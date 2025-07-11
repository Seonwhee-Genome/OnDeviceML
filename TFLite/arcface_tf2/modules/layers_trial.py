import tensorflow as tf
import math


class BatchNormalization(tf.keras.layers.BatchNormalization):
    """Make trainable=False freeze BN for real (the og version is sad).
       ref: https://github.com/zzh8829/yolov3-tf2
    """
#     def __init__(self, **kwargs):
#         super().__init__(epsilon=1e-2, **kwargs)
    
    def call(self, x, training=False):
        if training is None:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


class ArcMarginPenaltyLogists(tf.keras.layers.Layer):
    """ArcMarginPenaltyLogists"""
    def __init__(self, num_classes, margin=0.5, logist_scale=64, **kwargs):
        super(ArcMarginPenaltyLogists, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.margin = margin
        self.logist_scale = logist_scale

    def build(self, input_shape):
        self.w = self.add_variable(
            "weights", shape=[int(input_shape[-1]), self.num_classes])
        self.cos_m = tf.identity(math.cos(self.margin), name='cos_m')
        self.sin_m = tf.identity(math.sin(self.margin), name='sin_m')
        self.th = tf.identity(math.cos(math.pi - self.margin), name='th')
        self.mm = tf.multiply(self.sin_m, self.margin, name='mm')

    def call(self, embds, labels):
#         embds = tf.debugging.check_numerics(embds, "NaN or Inf in input embeddings")
#         labels = tf.debugging.check_numerics(tf.cast(labels, tf.float32), "NaN or Inf in labels")
        
        normed_embds = tf.nn.l2_normalize(embds, axis=1, epsilon=1e-6, name='normed_embd')
        normed_w = tf.nn.l2_normalize(self.w, axis=0, epsilon=1e-6, name='normed_weights')

        cos_t = tf.matmul(normed_embds, normed_w, name='cos_t')
#         sin_t = tf.sqrt(1. - cos_t ** 2, name='sin_t') ###
        cos_t = tf.clip_by_value(cos_t, -1.0 + 1e-5, 1.0 - 1e-5)
        tf.debugging.check_numerics(cos_t, "NaN in cos_t")
        sin_t = tf.sqrt(1. - tf.square(cos_t), name='sin_t')
        tf.debugging.check_numerics(sin_t, "NaN in sin_t")


        cos_mt = tf.subtract(
            cos_t * self.cos_m, sin_t * self.sin_m, name='cos_mt')

        cos_mt = tf.where(cos_t > self.th, cos_mt, cos_t - self.mm)

        mask = tf.one_hot(tf.cast(labels, tf.int32), depth=self.num_classes,
                          name='one_hot_mask')

        logists = tf.where(mask == 1., cos_mt, cos_t)
        logists = tf.multiply(logists, self.logist_scale, 'arcface_logist')

        return logists
