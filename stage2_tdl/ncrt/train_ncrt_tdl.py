"""NCRT: Noise-Conditioned Residual Transformer -- the transformer version of NCRN.

Identical to stage2_tdl/ncrn/train_ncrn_tdl.py in everything except the network:
same (T,64,64,3) input with SNR channel, same residual target h[-1]-hn[-1], same
3-phase SNR curriculum (make_dataset is imported), batch 4, 30 x 50 steps,
Adam 3e-4, ReduceLROnPlateau + EarlyStopping on loss, MSE loss. The only training
addition is clipnorm=1.0 (standard for attention models).

  python train_ncrt_tdl.py                    # -> models/ncrt_tdl_64.keras
  python train_ncrt_tdl.py --epochs 2 --tag _smoke
"""
import os, sys, time, pickle, argparse

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(HERE, '..', 'ncrn'))
sys.path.insert(0, HERE)

from shared.channels import TIME_STEPS
from train_ncrn_tdl import make_dataset, EPOCHS, STEPS_EPOCH
from ncrt_layers import (RMSScaleIn, RMSScaleOut, SNRToken, NoiseCondTemporalBlock,
                         NoiseCondReadout, AxialBlock)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
L = tf.keras.layers

if __name__ == '__main__' and '--gpu' not in sys.argv:   # must precede any tf.Variable
    tf.config.set_visible_devices([], 'GPU')

SKIPPED = tf.Variable(0.0, trainable=False)


class SkipNonFinite(tf.keras.Model):
    """Functional model whose train step drops batches with non-finite gradients.
    tensorflow-metal sporadically returns NaN conv gradients on the (B*T,64,64,C)
    batches; with per-variable clipnorm one such step silently poisons a few
    weights for good, and in a long-running process the NaNs can become permanent.
    Training therefore defaults to CPU (also faster for this model on an M1:
    1.1 vs 1.45 s/step); this guard stays as a safety net. Skipped batches are
    counted in the 'skipped' log."""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.loss_mean = tf.keras.metrics.Mean(name='loss')

    @property
    def metrics(self):
        return [self.loss_mean] + super().metrics

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x=x, y=y, y_pred=y_pred)
        grads = tape.gradient(loss, self.trainable_variables)
        ok = tf.reduce_all([tf.reduce_all(tf.math.is_finite(g)) for g in grads])
        SKIPPED.assign_add(1.0 - tf.cast(ok, tf.float32))
        self.optimizer.apply([tf.where(ok, g, tf.zeros_like(g)) for g in grads],
                             self.trainable_variables)
        self.loss_mean.update_state(loss, sample_weight=tf.cast(ok, tf.float32))
        # NaN (not 0) if every batch so far was skipped, so callbacks never see a fake best
        loss_log = tf.where(self.loss_mean.count > 0, self.loss_mean.result(), float('nan'))
        return {**self.compute_metrics(x, y, y_pred), 'loss': loss_log,
                'skipped': SKIPPED.value()}


def build_ncrt(Nr, Nt, T=TIME_STEPS, dim=64, heads=4, ffn=128, blocks=1,
               sp_dim=128, sp_blocks=2, ax_ffn=256, k_dw=7):
    """Temporal stage runs on all T frames (memory-bound, kept at dim=64, 1 block
    + the readout = 2 temporal attentions); the array-wide axial stage runs on the
    last frame only, so it gets the width. 888k params vs NCRN's 1.20M."""
    inp = tf.keras.Input(shape=(T, Nr, Nt, 3))
    x, s = RMSScaleIn()(inp)

    # Noise embedding (NCRN's SNR channel, lifted to a conditioning vector)
    c = L.Dense(64, activation='gelu')(SNRToken()(inp))
    c = L.Dense(64, activation='gelu')(c)

    # ConvLSTM2D(128, return_sequences=True)  ->  temporal transformer
    x = L.TimeDistributed(L.Conv2D(dim, (3, 3), padding='same'))(x)
    x1 = None
    for i in range(blocks):
        x = NoiseCondTemporalBlock(dim, heads, ffn, k_dw=k_dw, T=T)(x, c)
        if i == 0:
            x1 = x
    # ConvLSTM2D(64, return_sequences=False)  ->  last-frame readout + array-wide attention
    x2 = NoiseCondReadout(dim, heads, T=T)(x, c)
    x2 = L.Conv2D(sp_dim, (1, 1))(x2)
    for _ in range(sp_blocks):
        x2 = AxialBlock(sp_dim, heads, ax_ffn, n=Nt)(x2, c)
    x2 = L.Conv2D(64, (1, 1))(L.LayerNormalization()(x2))

    # 1x1 residual skip from last frame of layer 1 (as in NCRN)
    sk = L.Conv2D(64, (1, 1))(x1[:, -1])
    mg = L.Activation('gelu')(L.Add()([x2, sk]))

    # Conv head 128/64/2 (as in NCRN; GELU instead of ReLU so it cannot die to 0)
    out = L.Conv2D(128, (3, 3), padding='same', activation='gelu')(mg)
    out = L.Conv2D(64, (3, 3), padding='same', activation='gelu')(out)
    out = L.Conv2D(2, (3, 3), padding='same', activation='linear')(out)
    out = RMSScaleOut()(out, s)
    return tf.keras.Model(inp, out, name='ncrt')


def main():
    ap = argparse.ArgumentParser()
    # bs 2 x accum 2 x 100 steps == NCRN's bs 4 x 50 updates per epoch
    # (bs 4 swaps on an 8 GB M1). Same samples, same updates, same curriculum.
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--bs', type=int, default=2)
    ap.add_argument('--accum', type=int, default=2)
    ap.add_argument('--steps', type=int, default=None, help='default: STEPS_EPOCH*4/bs')
    ap.add_argument('--dim', type=int, default=64)
    ap.add_argument('--blocks', type=int, default=1)
    ap.add_argument('--sp_dim', type=int, default=128)
    ap.add_argument('--sp_blocks', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tag', default='')
    ap.add_argument('--gpu', action='store_true', help='use tensorflow-metal GPU (see SkipNonFinite)')
    ap.add_argument('--matlab_data', default='', help='path to MATLAB .mat channel file')
    ap.add_argument('--weights', default='', help='optional path to initial weights to load')
    a = ap.parse_args()

    np.random.seed(a.seed); tf.random.set_seed(a.seed)
    a.steps = a.steps or STEPS_EPOCH * 4 // a.bs
    Nr = Nt = 64
    print(f"\n{'='*64}")
    print(f"  NCRT Stage II — TDL (10 scenarios), {Nr}x{Nt}")
    print(f"  Curriculum: [5,15] -> [0,15] -> [-5,15] dB | Epochs: {a.epochs}")
    print(f"  Target: residual | Input: 3 channels (real, imag, SNR)")
    print(f"{'='*64}")

    net = build_ncrt(Nr, Nt, dim=a.dim, blocks=a.blocks, sp_dim=a.sp_dim, sp_blocks=a.sp_blocks)
    model = SkipNonFinite(net.inputs, net.outputs)
    opt = tf.keras.optimizers.Adam(3e-4, clipnorm=1.0,
                                   gradient_accumulation_steps=a.accum if a.accum > 1 else None)
    model.compile(optimizer=opt, loss='mse', metrics=['mae'])
    if a.weights:
        print(f"  Loading pre-trained weights from {a.weights}")
        if a.weights.endswith('.keras'):
            pre = tf.keras.models.load_model(a.weights, compile=False)
            net.set_weights(pre.get_weights())
        else:
            model.load_weights(a.weights)
    model.summary()
    print(f"\n  Params: {model.count_params():,}  (NCRN: 1,204,610)\n")

    if a.matlab_data:
        from shared.matlab_channels import load_matlab_6g_channels, make_matlab_dataset
        print(f"  Loading MATLAB dataset from: {a.matlab_data}")
        channels = load_matlab_6g_channels(a.matlab_data, target_Nr=Nr, target_Nt=Nt)
        ds = make_matlab_dataset(channels, batch_size=a.bs, total_epochs=a.epochs,
                                 steps_per_epoch=a.steps)
    else:
        ds = make_dataset(Nr, Nt, a.bs, a.epochs, a.steps)
    out_dir = os.path.join(ROOT, 'models'); os.makedirs(out_dir, exist_ok=True)
    name = f'ncrt_tdl_64{a.tag}'
    cbs = [tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=6,
                                                min_lr=1e-6, verbose=1),
           tf.keras.callbacks.EarlyStopping(monitor='loss', patience=15,
                                            restore_best_weights=True, verbose=1),
           tf.keras.callbacks.ModelCheckpoint(os.path.join(out_dir, name + '_best.weights.h5'),
                                              monitor='loss', save_best_only=True,
                                              save_weights_only=True)]
    t0 = time.time()
    his = model.fit(ds, steps_per_epoch=a.steps, epochs=a.epochs, callbacks=cbs, verbose=2)

    net.save(os.path.join(out_dir, name + '.keras'))     # plain functional, shares weights
    with open(os.path.join(out_dir, name + '_history.pkl'), 'wb') as f:
        pickle.dump({'loss': his.history['loss'], 'mae': his.history['mae'],
                     'skipped': his.history['skipped']}, f)
    print(f"\n  Done in {time.time()-t0:.1f}s -> models/{name}.keras "
          f"({int(SKIPPED.numpy())} non-finite batches skipped)")


if __name__ == '__main__':
    main()
