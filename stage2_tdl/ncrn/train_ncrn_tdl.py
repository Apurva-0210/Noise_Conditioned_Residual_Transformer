import os, sys, time, pickle, argparse

# Windows' cmd/PowerShell consoles default to a legacy codepage that can't
# encode the box-drawing/unicode characters used in this script's output
# (UnicodeEncodeError). Force UTF-8 stdout/stderr on Windows only.
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
import numpy as np
import tensorflow as tf
if '--gpu' not in sys.argv:
    tf.config.set_visible_devices([], 'GPU')

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                  '..', '..'))
from shared.channels  import gen_tdl_random, TIME_STEPS
from shared.baselines import add_snr_channel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# ─── Config ───────────────────────────────────────────────────
EPOCHS      = 30
STEPS_EPOCH = 50


# ─── Custom layer (required for loading trained model) ────────
class LastFrame(tf.keras.layers.Layer):
    def call(self, x): return x[:, -1, :, :, :]
    def get_config(self): return super().get_config()


def build_ncrn(Nr, Nt):
    """Wide ConvLSTM 128/64 + 1x1 residual skip + Conv head 128/64/2."""
    inp = tf.keras.Input(shape=(TIME_STEPS, Nr, Nt, 3))   # 3 ch w/ SNR
    x1  = tf.keras.layers.ConvLSTM2D(
              128, (3,3), padding='same', return_sequences=True,
              activation='tanh', recurrent_activation='sigmoid',
              kernel_regularizer=tf.keras.regularizers.l2(1e-5))(inp)
    x1  = tf.keras.layers.BatchNormalization()(x1)
    x2  = tf.keras.layers.ConvLSTM2D(
              64, (3,3), padding='same', return_sequences=False,
              activation='tanh', recurrent_activation='sigmoid',
              kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x1)
    x2  = tf.keras.layers.BatchNormalization()(x2)
    sk  = LastFrame()(x1)
    sk  = tf.keras.layers.Conv2D(64, (1,1), padding='same')(sk)
    mg  = tf.keras.layers.Add()([x2, sk])
    mg  = tf.keras.layers.Activation('relu')(mg)
    out = tf.keras.layers.Conv2D(128, (3,3), padding='same',
                                  activation='relu')(mg)
    out = tf.keras.layers.Conv2D(64,  (3,3), padding='same',
                                  activation='relu')(out)
    out = tf.keras.layers.Conv2D(2,   (3,3), padding='same',
                                  activation='linear')(out)
    m = tf.keras.Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(3e-4),
              loss='mse', metrics=['mae'])
    return m


def make_dataset(Nr, Nt, batch_size, total_epochs, steps_per_epoch):
    """Three-phase curriculum on random TDL scenarios."""
    total = total_epochs * steps_per_epoch
    step  = [0]
    def _gen():
        while True:
            prog = step[0] / total
            snr_min = (5.0 if prog < 0.33 else
                       0.0 if prog < 0.67 else -5.0)
            snr_max = 15.0
            X, Y = [], []
            for _ in range(batch_size):
                h = gen_tdl_random(Nr, Nt)
                snr_db = float(np.random.uniform(snr_min, snr_max))
                sp     = float(np.mean(h**2)) + 1e-10
                ns     = np.sqrt(sp * 10**(-snr_db/10))
                hn     = (h + np.random.normal(0, ns, h.shape)).astype(np.float32)
                hn_cond = add_snr_channel(hn, snr_db)
                target  = (h[-1] - hn[-1]).astype(np.float32)   # residual
                X.append(hn_cond); Y.append(target)
            step[0] += 1
            yield np.array(X, dtype=np.float32), \
                  np.array(Y, dtype=np.float32)
    return tf.data.Dataset.from_generator(
        _gen,
        output_signature=(
            tf.TensorSpec((batch_size, TIME_STEPS, Nr, Nt, 3), tf.float32),
            tf.TensorSpec((batch_size, Nr, Nt, 2),             tf.float32),
        )
    ).prefetch(tf.data.AUTOTUNE)


def main(Nr=64, Nt=64):
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--steps', type=int, default=STEPS_EPOCH)
    ap.add_argument('--bs', type=int, default=4)
    ap.add_argument('--tag', default='')
    ap.add_argument('--matlab_data', default='', help='path to MATLAB .mat channel file')
    ap.add_argument('--weights', default='', help='path to pre-trained weights/model')
    a = ap.parse_args()

    suffix     = '128' if Nt == 128 else '64'
    batch_size = a.bs

    print(f"\n{'='*64}")
    print(f"  NCRN Stage II — TDL (10 scenarios), {Nr}x{Nt}")
    print(f"  Curriculum: [5,15] -> [0,15] -> [-5,15] dB | Epochs: {a.epochs}")
    print(f"  Target: residual | Input: 3 channels (real, imag, SNR)")
    print(f"{'='*64}")

    model = build_ncrn(Nr, Nt)
    if a.weights:
        print(f"  Loading pre-trained weights from: {a.weights}")
        if a.weights.endswith('.keras'):
            pre = tf.keras.models.load_model(a.weights, custom_objects={'LastFrame': LastFrame}, compile=False)
            model.set_weights(pre.get_weights())
        else:
            model.load_weights(a.weights)
    model.summary()
    print(f"\n  Params: {model.count_params():,}\n")

    if a.matlab_data:
        from shared.matlab_channels import load_matlab_6g_channels, make_matlab_dataset
        print(f"  Loading MATLAB dataset from: {a.matlab_data}")
        channels = load_matlab_6g_channels(a.matlab_data, target_Nr=Nr, target_Nt=Nt)
        ds = make_matlab_dataset(channels, batch_size=batch_size, total_epochs=a.epochs,
                                 steps_per_epoch=a.steps)
    else:
        ds  = make_dataset(Nr, Nt, batch_size, a.epochs, a.steps)
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    os.makedirs(out_dir, exist_ok=True)
    out_path  = os.path.join(out_dir, f'ncrn_tdl_{suffix}{a.tag}.keras')
    hist_path = os.path.join(out_dir, f'ncrn_tdl_{suffix}{a.tag}_history.pkl')

    lr  = tf.keras.callbacks.ReduceLROnPlateau(
              monitor='loss', factor=0.5, patience=4,
              min_lr=1e-6, verbose=1)
    es  = tf.keras.callbacks.EarlyStopping(
              monitor='loss', patience=8,
              restore_best_weights=True, verbose=1)
    ckpt = tf.keras.callbacks.ModelCheckpoint(
              out_path, monitor='loss', save_best_only=True, verbose=1)

    t0  = time.time()
    his = model.fit(ds, steps_per_epoch=a.steps, epochs=a.epochs,
                    callbacks=[lr, es, ckpt], verbose=2)
    elapsed = time.time() - t0

    model.save(out_path)
    with open(hist_path, 'wb') as f:
        pickle.dump({'loss': his.history['loss'],
                      'mae':  his.history['mae']}, f)
    print(f"\n  Done in {elapsed:.1f}s")
    print(f"  Model:   {out_path}")


if __name__ == '__main__':
    main(64, 64)
