"""Build -> 2 train steps -> save -> reload -> predict -> param count. Run before any long job."""
import os, sys, tempfile
import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train_trust_net import build_trust_net, make_loss
from train_ncrn_tdl import build_ncrn, LastFrame
import trust_layers  # noqa: registers layers


def main():
    print(f"NCRN params (64x64): {build_ncrn(64, 64).count_params():,}")
    for v in ('a1', 'a2'):
        n = build_trust_net(64, 64, variant=v).count_params()
        print(f"TRUST-Net {v} params (64x64): {n:,}  ({n/1_204_610-1:+.1%} vs NCRN)")

        Nr = Nt = 16
        m = build_trust_net(Nr, Nt, variant=v)
        loss, metrics = make_loss(v)
        m.compile(optimizer=tf.keras.optimizers.Adam(3e-4), loss=loss, metrics=metrics)
        x = np.random.randn(2, 10, Nr, Nt, 3).astype('f')
        y = np.random.randn(2, Nr, Nt, 2).astype('f')
        for _ in range(2):
            l = m.train_on_batch(x, y)
        assert np.all(np.isfinite(l)), f"{v}: non-finite loss {l}"
        p0 = m.predict(x, verbose=0)
        path = os.path.join(tempfile.mkdtemp(), f'{v}.keras')
        m.save(path)
        m2 = tf.keras.models.load_model(path, compile=False,
                                          custom_objects={'LastFrame': LastFrame})
        p1 = m2.predict(x, verbose=0)
        np.testing.assert_allclose(p0, p1, atol=1e-4)
        assert p1.shape[-1] == (2 if v == 'a1' else 8)
        print(f"  {v}: loss={np.ravel(l)[0]:.4f}, reload OK, out {p1.shape}")
    print("SMOKE TEST PASSED")


if __name__ == '__main__':
    main()
