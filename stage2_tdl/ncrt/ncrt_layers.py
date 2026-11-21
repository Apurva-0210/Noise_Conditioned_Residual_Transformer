"""Layers for NCRT (Noise-Conditioned Residual Transformer).

One-to-one transformer counterparts of NCRN's components:

  NCRN                                  NCRT
  ----------------------------------    ------------------------------------------
  SNR as a 3rd input channel            same input + AdaLN-Zero: every norm is
                                        modulated by an SNR embedding
  ConvLSTM2D(128, seq=True)             NoiseCondTemporalBlock x N (temporal MHSA
                                        with relative bias + depthwise ConvFFN)
  ConvLSTM2D(64, seq=False)             NoiseCondReadout (last-frame query
                                        cross-attends over all T frames)
  3x3 receptive field (local)           AxialBlock: row + column attention over the
                                        full 64-antenna ULA with |i-j| bias
  BatchNorm                             RMSScaleIn / RMSScaleOut: exact per-sample
                                        scale equivariance (TDL power spans ~30x)

Temporal attention / ConvFFN are reused from trust_net/trust_layers.py.
"""
import os, sys
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'trust_net'))
from trust_layers import TemporalAttention, ConvFFN  # noqa: E402

L = tf.keras.layers
reg = tf.keras.utils.register_keras_serializable(package='ncrt')


@reg
class RMSScaleIn(L.Layer):
    """(B,T,H,W,3) -> (normalised input, per-sample RMS s of shape (B,1,1,1)).
    s comes from the noisy observation only, so it is available at inference."""
    def call(self, x):
        s = tf.sqrt(tf.reduce_mean(tf.square(x[..., :2]), axis=[1, 2, 3, 4], keepdims=True) + 1e-8)
        return tf.concat([x[..., :2] / s, x[..., 2:]], -1), s[:, 0]


@reg
class RMSScaleOut(L.Layer):
    def call(self, y, s):
        return y * s


@reg
class SNRToken(L.Layer):
    """Reads the normalised SNR (channel 2, constant over the grid) -> (B,1)."""
    def call(self, x):
        return x[:, 0, 0, 0, 2:3]


@reg
class AdaLN(L.Layer):
    """LayerNorm whose scale/shift/gate are regressed from the noise embedding c.
    Zero-initialised, so it starts as a plain LN with unit gate; the SNR then
    learns how strongly each branch should act (stronger smoothing at low SNR)."""
    def __init__(self, dim, **kw):
        super().__init__(**kw)
        self.dim = dim
        self.ln = L.LayerNormalization(center=False, scale=False, epsilon=1e-6)
        self.mod = L.Dense(3 * dim, kernel_initializer='zeros', bias_initializer='zeros')

    def call(self, x, c):
        m = self.mod(c)                                    # (B, 3C)
        m = tf.reshape(m, tf.concat([[tf.shape(x)[0]], tf.ones([tf.rank(x) - 2], tf.int32),
                                     [3 * self.dim]], 0))
        scale, shift, gate = tf.split(m, 3, axis=-1)
        return self.ln(x) * (1.0 + scale) + shift, gate

    def get_config(self):
        return {**super().get_config(), 'dim': self.dim}


@reg
class NoiseCondTemporalBlock(L.Layer):
    """(B,T,H,W,C): x + g1*MHSA_t(AdaLN x); x + g2*ConvFFN(AdaLN x)."""
    def __init__(self, dim, heads=4, ffn=256, k_dw=7, T=10, **kw):
        super().__init__(**kw)
        self.dim, self.heads, self.ffn, self.k_dw, self.T = dim, heads, ffn, k_dw, T
        self.n1, self.n2 = AdaLN(dim), AdaLN(dim)
        self.attn = TemporalAttention(dim, heads, last_only=False, T=T)
        self.mlp = ConvFFN(dim, ffn, k_dw=k_dw)

    def call(self, x, c):
        h, g = self.n1(x, c)
        x = x + (1.0 + g) * self.attn(h)
        h, g = self.n2(x, c)
        return x + (1.0 + g) * self.mlp(h)

    def get_config(self):
        return {**super().get_config(), 'dim': self.dim, 'heads': self.heads,
                'ffn': self.ffn, 'k_dw': self.k_dw, 'T': self.T}


@reg
class NoiseCondReadout(L.Layer):
    """Replaces ConvLSTM2D(return_sequences=False): (B,T,H,W,C) -> (B,H,W,C)."""
    def __init__(self, dim, heads=4, T=10, **kw):
        super().__init__(**kw)
        self.dim, self.heads, self.T = dim, heads, T
        self.n = AdaLN(dim)
        self.attn = TemporalAttention(dim, heads, last_only=True, T=T)

    def call(self, x, c):
        h, g = self.n(x, c)
        return x[:, -1] + (1.0 + g[:, 0]) * self.attn(h)

    def get_config(self):
        return {**super().get_config(), 'dim': self.dim, 'heads': self.heads, 'T': self.T}


@reg
class AxialAttention(L.Layer):
    """MHSA along one antenna axis of (B,H,W,C), with a learned bias per |i-j|
    offset (the ULA correlation rho^|i-j| depends only on that offset)."""
    def __init__(self, dim, heads=4, axis='row', n=64, **kw):
        super().__init__(**kw)
        self.dim, self.heads, self.axis, self.n = dim, heads, axis, n
        self.hd = dim // heads
        self.qkv = L.Dense(3 * dim)
        self.o = L.Dense(dim)

    def build(self, shape):
        self.rel_bias = self.add_weight(name='rel_bias', shape=(self.heads, 2 * self.n - 1),
                                        initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))
        super().build(shape)

    def call(self, x):
        B, H, W = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        q, k, v = tf.split(self.qkv(x), 3, axis=-1)
        split = lambda t: tf.reshape(t, [B, H, W, self.heads, self.hd])
        q, k, v = split(q), split(k), split(v)
        # Toeplitz bias via a dense one-hot contraction: a tf.gather here gives a
        # sparse gradient that is ~10x slower on Metal than the whole model.
        idx = tf.range(self.n)
        onehot = tf.one_hot(idx[:, None] - idx[None, :] + (self.n - 1), 2 * self.n - 1)
        bias = tf.einsum('hk,ijk->hij', self.rel_bias, onehot)       # (heads,N,N)
        if self.axis == 'row':                                         # attend along W
            s = tf.einsum('bhwnd,bhvnd->bhnwv', q, k) * self.hd ** -0.5 + bias[None, None]
            y = tf.einsum('bhnwv,bhvnd->bhwnd', tf.nn.softmax(s, -1), v)
        else:                                                          # attend along H
            s = tf.einsum('bhwnd,bgwnd->bwnhg', q, k) * self.hd ** -0.5 + bias[None, None]
            y = tf.einsum('bwnhg,bgwnd->bhwnd', tf.nn.softmax(s, -1), v)
        return self.o(tf.reshape(y, [B, H, W, self.dim]))

    def get_config(self):
        return {**super().get_config(), 'dim': self.dim, 'heads': self.heads,
                'axis': self.axis, 'n': self.n}


@reg
class AxialBlock(L.Layer):
    """(B,H,W,C): row attn, column attn, then 1x1 -> dw3x3 -> 1x1 FFN, all noise-conditioned."""
    def __init__(self, dim, heads=4, ffn=192, n=64, **kw):
        super().__init__(**kw)
        self.dim, self.heads, self.ffn, self.n = dim, heads, ffn, n
        self.n1, self.n2, self.n3 = AdaLN(dim), AdaLN(dim), AdaLN(dim)
        self.row = AxialAttention(dim, heads, 'row', n)
        self.col = AxialAttention(dim, heads, 'col', n)
        self.f1 = L.Conv2D(ffn, 1)
        self.dw = L.DepthwiseConv2D(3, padding='same')
        self.f2 = L.Conv2D(dim, 1)

    def call(self, x, c):
        h, g = self.n1(x, c); x = x + (1.0 + g) * self.row(h)
        h, g = self.n2(x, c); x = x + (1.0 + g) * self.col(h)
        h, g = self.n3(x, c)
        return x + (1.0 + g) * self.f2(tf.nn.gelu(self.dw(tf.nn.gelu(self.f1(h)))))

    def get_config(self):
        return {**super().get_config(), 'dim': self.dim, 'heads': self.heads,
                'ffn': self.ffn, 'n': self.n}
