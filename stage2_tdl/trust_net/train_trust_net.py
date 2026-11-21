"""TRUST-Net: NCRN with ConvLSTM2D swapped for temporal self-attention.

Same input / residual target / curriculum / budget / callbacks as
stage2_tdl/ncrn/train_ncrn_tdl.py (its make_dataset and LastFrame are reused).

  python train_trust_net.py --variant a1 --seed 0
  python train_trust_net.py --variant a2 --seed 0      # evidential head
"""
import os, sys, time, pickle, argparse
import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(HERE, '..', 'ncrn'))
sys.path.insert(0, HERE)

from shared.channels import TIME_STEPS
from shared.evidential import split_nig_params, der_h_loss, der_h_mae
from shared.matlab_channels import load_matlab_6g_channels, make_matlab_dataset
from train_ncrn_tdl import LastFrame, make_dataset, EPOCHS, STEPS_EPOCH
from trust_layers import TemporalPosEmbed, TemporalBlock, LastFrameReadout

L = tf.keras.layers


def build_trust_net(Nr, Nt, T=TIME_STEPS, variant='a1', dim=128, heads=4,
                    ffn=312, blocks=2, pos_emb=True, k_dw=7):
    inp = tf.keras.Input(shape=(T, Nr, Nt, 3))
    x = L.TimeDistributed(L.Conv2D(dim, (3, 3), padding='same'))(inp)      # shared per-frame embed
    if pos_emb:
        x = TemporalPosEmbed(T)(x)
    x1 = None
    for i in range(blocks):
        x = TemporalBlock(dim, heads, ffn, k_dw=k_dw, T=T)(x)
        if i == 0:
            x1 = x
    x2 = LastFrameReadout(dim, heads, T=T)(x)
    x2 = L.Conv2D(64, (1, 1), padding='same')(x2)
    x2 = L.BatchNormalization()(x2)
    sk = L.Conv2D(64, (1, 1), padding='same')(LastFrame()(x1))
    mg = L.Activation('relu')(L.Add()([x2, sk]))
    out = L.Conv2D(128, (3, 3), padding='same', activation='relu')(mg)
    out = L.Conv2D(64, (3, 3), padding='same', activation='relu')(out)
    out = L.Conv2D(2 if variant == 'a1' else 8, (3, 3), padding='same',
                   activation='linear')(out)
    return tf.keras.Model(inp, out)


def make_loss(variant, lam=0.1, mu=1.0):
    if variant == 'a1':
        return 'mse', ['mae']
    der = der_h_loss(coeff_reg=lam)

    def loss(y, p):
        # gamma_re / gamma_im sit at channels 0 and 4 (raw, linear)
        gamma = tf.stack([p[..., 0], p[..., 4]], axis=-1)
        return der(y, p) + mu * tf.reduce_mean(tf.square(y - gamma))
    return loss, [der_h_mae]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=['a1', 'a2'], default='a1')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--dim', type=int, default=128)
    ap.add_argument('--ffn', type=int, default=312)
    ap.add_argument('--k_dw', type=int, default=7, help='kernel size for depthwise spatial conv')
    ap.add_argument('--blocks', type=int, default=2)
    ap.add_argument('--T', type=int, default=TIME_STEPS, help='use last T frames')
    ap.add_argument('--no_pos', action='store_true')
    ap.add_argument('--lam', type=float, default=0.1)
    ap.add_argument('--mu', type=float, default=1.0)
    ap.add_argument('--bs', type=int, default=1, help='batch size (default 1 to fit in memory)')
    ap.add_argument('--grad_accum', type=int, default=4, help='gradient accumulation steps (default 4 to match bs=4)')
    ap.add_argument('--epochs', type=int, default=EPOCHS, help='total epochs (default 30)')
    ap.add_argument('--steps_per_epoch', type=int, default=200, help='steps per epoch (default 200 with bs=1 matches 50 with bs=4)')
    ap.add_argument('--no_early_stopping', action='store_true', help='train all epochs without premature early stop')
    ap.add_argument('--matlab_data', default='', help='path to MATLAB .mat 6G channel file')
    ap.add_argument('--tag', default='')
    a = ap.parse_args()

    np.random.seed(a.seed); tf.random.set_seed(a.seed)
    Nr = Nt = 64
    bs = a.bs
    epochs = a.epochs
    steps_per_epoch = a.steps_per_epoch
    model = build_trust_net(Nr, Nt, a.T, a.variant, a.dim, 4, a.ffn, a.blocks, not a.no_pos, k_dw=a.k_dw)
    loss, metrics = make_loss(a.variant, a.lam, a.mu)

    if a.grad_accum > 1:
        opt = tf.keras.optimizers.Adam(3e-4, gradient_accumulation_steps=a.grad_accum)
    else:
        opt = tf.keras.optimizers.Adam(3e-4)

    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    model.summary()
    print(f"\n  Params: {model.count_params():,}  (NCRN: 1,204,610)\n")

    if a.matlab_data:
        print(f"  [MATLAB 6G] Loading channel data from: {a.matlab_data}")
        mat_channels = load_matlab_6g_channels(a.matlab_data, target_T=a.T, target_Nr=Nr, target_Nt=Nt)
        ds = make_matlab_dataset(mat_channels, bs, epochs, steps_per_epoch)
    else:
        ds = make_dataset(Nr, Nt, bs, epochs, steps_per_epoch)
    if a.T != TIME_STEPS:
        ds = ds.map(lambda x, y: (x[:, -a.T:], y))
    
    cbs = [tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=6,
                                                min_lr=1e-6, verbose=1)]
    out_dir = os.path.join(ROOT, 'models'); os.makedirs(out_dir, exist_ok=True)
    name = f'trust_net_{a.variant}{a.tag}_seed{a.seed}'
    ckpt_path = os.path.join(out_dir, name + '_ckpt.keras')
    cbs.append(tf.keras.callbacks.ModelCheckpoint(ckpt_path, save_weights_only=False, verbose=1))
    if not a.no_early_stopping:
        cbs.append(tf.keras.callbacks.EarlyStopping(monitor='loss', patience=15,
                                                    restore_best_weights=True, verbose=1))
    t0 = time.time()
    his = model.fit(ds, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=cbs, verbose=1)

    out_dir = os.path.join(ROOT, 'models'); os.makedirs(out_dir, exist_ok=True)
    name = f'trust_net_{a.variant}{a.tag}_seed{a.seed}'
    model.save(os.path.join(out_dir, name + '.keras'))
    with open(os.path.join(out_dir, name + '_history.pkl'), 'wb') as f:
        pickle.dump({k: v for k, v in his.history.items()}, f)
    print(f"\n  Done in {time.time()-t0:.1f}s -> models/{name}.keras")
    print("  (a2 models: load with compile=False)")


if __name__ == '__main__':
    main()
