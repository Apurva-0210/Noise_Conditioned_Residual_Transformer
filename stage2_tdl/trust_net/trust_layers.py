"""Temporal-attention layers for TRUST-Net.

Attention runs over the time axis only, independently at every (Nr, Nt)
antenna location. Scores are (B, H, W, heads, Tq, Tk), which is tiny; the
Keras MultiHeadAttention layer is avoided because it OOM'd on the folded
B*H*W sequence batch.
"""
import tensorflow as tf

L = tf.keras.layers
reg = tf.keras.utils.register_keras_serializable(package='trust')


@reg
class TemporalPosEmbed(L.Layer):
    """Adds a learned (T, C) embedding to a (B, T, H, W, C) tensor."""
    def __init__(self, T, **kw):
        super().__init__(**kw)
        self.T = T

    def build(self, shape):
        self.pos = self.add_weight(name='pos', shape=(self.T, shape[-1]),
                                   initializer=tf.keras.initializers.RandomNormal(stddev=0.02))

    def call(self, x):
        return x + self.pos[None, :, None, None, :]

    def get_config(self):
        return {**super().get_config(), 'T': self.T}


@reg
class TemporalAttention(L.Layer):
    """Multi-head self-attention over T at each spatial location with relative temporal bias.

    last_only=False: (B,T,H,W,C) -> (B,T,H,W,C)
    last_only=True : queries come from the last frame only -> (B,H,W,C)
    """
    def __init__(self, dim, heads=4, last_only=False, T=10, **kw):
        super().__init__(**kw)
        assert dim % heads == 0
        self.dim, self.heads, self.last_only, self.T = dim, heads, last_only, T
        self.hd = dim // heads
        self.q = L.Dense(dim)
        self.k = L.Dense(dim)
        self.v = L.Dense(dim)
        self.o = L.Dense(dim)

    def build(self, input_shape):
        self.rel_bias = self.add_weight(
            name='rel_bias', shape=(self.heads, 2 * self.T - 1),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))
        self.q.build(input_shape)
        self.k.build(input_shape)
        self.v.build(input_shape)
        self.o.build(input_shape)
        super().build(input_shape)

    def _split(self, t):                       # (...,C) -> (...,heads,hd)
        return tf.reshape(t, tf.concat([tf.shape(t)[:-1], [self.heads, self.hd]], 0))

    def call(self, x):
        xq = x[:, -1:] if self.last_only else x            # (B,Tq,H,W,C)
        q, k, v = self._split(self.q(xq)), self._split(self.k(x)), self._split(self.v(x))
        s = tf.einsum('bthwnd,bshwnd->bhwnts', q, k) * (self.hd ** -0.5)

        t_q = tf.shape(q)[1]
        t_k = tf.shape(k)[1]
        if self.last_only:
            dist = (self.T - 1) - tf.range(t_k) + (self.T - 1)
            bias = tf.gather(self.rel_bias, dist, axis=-1)
            s = s + bias[None, None, None, :, None, :]
        else:
            q_idx = tf.range(t_q)[:, None]
            k_idx = tf.range(t_k)[None, :]
            dist = q_idx - k_idx + (self.T - 1)
            bias = tf.gather(self.rel_bias, dist, axis=-1)
            s = s + bias[None, None, None, :, :, :]

        a = tf.nn.softmax(s, axis=-1)
        y = tf.einsum('bhwnts,bshwnd->bthwnd', a, v)
        y = tf.reshape(y, tf.concat([tf.shape(y)[:-2], [self.dim]], 0))
        y = self.o(y)
        return y[:, 0] if self.last_only else y

    def get_config(self):
        return {**super().get_config(), 'dim': self.dim, 'heads': self.heads,
                'last_only': self.last_only, 'T': self.T}


@reg
class ConvFFN(L.Layer):
    """Multi-scale spatial mixing: 3x3 expand -> Depthwise 7x7 -> GELU -> 1x1 project."""
    def __init__(self, dim, hidden, k_dw=7, **kw):
        super().__init__(**kw)
        self.dim, self.hidden, self.k_dw = dim, hidden, k_dw
        self.c1 = L.TimeDistributed(L.Conv2D(hidden, (3, 3), padding='same'))
        self.dw = L.TimeDistributed(L.DepthwiseConv2D((k_dw, k_dw), padding='same'))
        self.c2 = L.TimeDistributed(L.Conv2D(dim, (1, 1)))

    def build(self, input_shape):
        self.c1.build(input_shape)
        hidden_shape = list(input_shape)
        hidden_shape[-1] = self.hidden
        self.dw.build(tuple(hidden_shape))
        self.c2.build(tuple(hidden_shape))
        super().build(input_shape)

    def call(self, x):
        h = tf.nn.gelu(self.c1(x))
        h = tf.nn.gelu(self.dw(h))
        return self.c2(h)

    def get_config(self):
        return {**super().get_config(), 'dim': self.dim, 'hidden': self.hidden,
                'k_dw': self.k_dw}


@reg
class TemporalBlock(L.Layer):
    """Pre-LN: x + MHSA(LN x); x + ConvFFN(LN x)."""
    def __init__(self, dim, heads=4, ffn=312, k_dw=7, T=10, **kw):
        super().__init__(**kw)
        self.dim, self.heads, self.ffn, self.k_dw, self.T = dim, heads, ffn, k_dw, T
        self.ln1, self.ln2 = L.LayerNormalization(), L.LayerNormalization()
        self.attn = TemporalAttention(dim, heads, last_only=False, T=T)
        self.mlp = ConvFFN(dim, ffn, k_dw=k_dw)

    def build(self, input_shape):
        self.ln1.build(input_shape)
        self.attn.build(input_shape)
        self.ln2.build(input_shape)
        self.mlp.build(input_shape)
        super().build(input_shape)

    def call(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))

    def get_config(self):
        return {**super().get_config(), 'dim': self.dim, 'heads': self.heads,
                'ffn': self.ffn, 'k_dw': self.k_dw, 'T': self.T}


@reg
class LastFrameReadout(L.Layer):
    """Query = last frame, keys/values = all frames; replaces return_sequences=False.
    Output (B,H,W,C) = last frame + cross-attention update with relative temporal bias."""
    def __init__(self, dim, heads=4, T=10, **kw):
        super().__init__(**kw)
        self.dim, self.heads, self.T = dim, heads, T
        self.ln = L.LayerNormalization()
        self.attn = TemporalAttention(dim, heads, last_only=True, T=T)

    def build(self, input_shape):
        self.ln.build(input_shape)
        self.attn.build(input_shape)
        super().build(input_shape)

    def call(self, x):
        return x[:, -1] + self.attn(self.ln(x))

    def get_config(self):
        return {**super().get_config(), 'dim': self.dim, 'heads': self.heads, 'T': self.T}
