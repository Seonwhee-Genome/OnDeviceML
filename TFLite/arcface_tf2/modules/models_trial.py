import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Flatten,
    Input,
)
from tensorflow.keras.applications import (
    MobileNetV2,
    ResNet50
)
from .layers import (
    BatchNormalization,
    ArcMarginPenaltyLogists
)
from tensorflow.keras.layers import Lambda
from tensorflow_addons.layers import GroupNormalization
from tensorflow.keras.layers import LayerNormalization

def _regularizer(weights_decay=5e-4):
    return tf.keras.regularizers.l2(weights_decay)


def Backbone(backbone_type='ResNet50', use_pretrain=True):
    """Backbone Model"""
    weights = None
    if use_pretrain:
        weights = 'imagenet'

    def backbone(x_in):
        if backbone_type == 'ResNet50':
            return ResNet50(input_shape=x_in.shape[1:], include_top=False,
                            weights=weights)(x_in)
        elif backbone_type == 'MobileNetV2':
            return MobileNetV2(input_shape=x_in.shape[1:], include_top=False,
                               weights=weights)(x_in)
        else:
            raise TypeError('backbone_type error!')
#         base.trainable = False
#         return base(x_in)
    return backbone


# def OutputLayer(embd_shape, w_decay=5e-4, name='OutputLayer'):
#     """Output Later"""
#     def output_layer(x_in):
#         x = inputs = Input(x_in.shape[1:])
#         x = BatchNormalization()(x)
#         x = tf.debugging.check_numerics(x, message="NaN after first BN")
#         x = Dropout(rate=0.5)(x)
#         x = Flatten()(x)
#         x = tf.debugging.check_numerics(x, message="NaN after Flatten")
        
#         x = Dense(embd_shape, kernel_regularizer=_regularizer(w_decay))(x)
#         x = tf.clip_by_value(x, -1e4, 1e4)
#         x = tf.debugging.check_numerics(x, message="NaN after Dense")
# #         x = BatchNormalization()(x)
#         x = tf.debugging.check_numerics(x, message="NaN after second BN")
#         return Model(inputs, x, name=name)(x_in)
#     return output_layer




def debug_print(x):
    tf.print("Dense output stats — min:", tf.reduce_min(x), "max:", tf.reduce_max(x))
    return x

def debug_variance(tensor):
    var = tf.math.reduce_variance(tensor)
    tf.print("Variance before BN:", var)
    return tensor

def safe_dense(x, units, w_decay):
    x = tf.keras.layers.Dense(
        units, kernel_regularizer=_regularizer(w_decay),
        kernel_initializer=tf.keras.initializers.HeNormal()
    )(x)
    x = tf.clip_by_value(x, -10.0, 10.0)  # limit values
    tf.debugging.check_numerics(x, message="After Dense")
    return x



def OutputLayer(embd_shape, w_decay=5e-4, name='OutputLayer'):
    def output_layer(x_in):
        x = inputs = Input(x_in.shape[1:], name='input_to_output_layer')

        # First BN
#         x = BatchNormalization(name='bn1')(x)
#         x = tf.debugging.check_numerics(x, "NaN after first BN")

        # Dropout
        x = Dropout(rate=0.5, name='dropout')(x)

        # Flatten
        # Check the input before Flatten
        tf.debugging.check_numerics(x, "Before Flatten")
        x = Flatten(name='flatten')(x)
        x = tf.debugging.check_numerics(x, "After Flatten")

        # Dense layer
        x = safe_dense(x, embd_shape, w_decay)
#         x = Dense(embd_shape, kernel_regularizer=_regularizer(w_decay), name='dense_embd')(x)
        
        
        
#         x = tf.debugging.check_numerics(x, "After Dense")

        # Inspect stats manually
#         tf.print("Dense output stats — min:", tf.reduce_min(x), "max:", tf.reduce_max(x), summarize=-1)
#         x = Lambda(debug_print)(x)
#         x = Lambda(debug_variance)(x)

        # Final BN
#         x = tf.clip_by_value(x, -20.0, 20.0)
#         x = LayerNormalization(epsilon=1e-3, name='ln2')(x)
#         x = GroupNormalization(groups=32, axis=-1)(x)  # requires tensorflow-addons

#         x = BatchNormalization(epsilon=1e-3, name='bn2', fused=False)(x)
#         x = tf.debugging.check_numerics(x, "After Second Normalization")

        return Model(inputs, x, name=name)(x_in)
    return output_layer


def ArcHead(num_classes, margin=0.5, logist_scale=64, name='ArcHead'):
    """Arc Head"""
    def arc_head(x_in, y_in):
        x = inputs1 = Input(x_in.shape[1:])
        y = Input(y_in.shape[1:])
        x = ArcMarginPenaltyLogists(num_classes=num_classes,
                                    margin=margin,
                                    logist_scale=logist_scale)(x, y)
        return Model((inputs1, y), x, name=name)((x_in, y_in))
    return arc_head


def NormHead(num_classes, w_decay=5e-4, name='NormHead'):
    """Norm Head"""
    def norm_head(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = Dense(num_classes, kernel_regularizer=_regularizer(w_decay))(x)
        return Model(inputs, x, name=name)(x_in)
    return norm_head


def ArcFaceModel(size=None, channels=3, num_classes=None, name='arcface_model',
                 margin=0.5, logist_scale=64, embd_shape=512,
                 head_type='ArcHead', backbone_type='ResNet50',
                 w_decay=5e-4, use_pretrain=True, training=False):
    """Arc Face Model"""
    x = inputs = Input([size, size, channels], name='input_image')

    x = Backbone(backbone_type=backbone_type, use_pretrain=use_pretrain)(x)
#     tf.print("Backbone output stats — min:", tf.reduce_min(x), "max:", tf.reduce_max(x), summarize=-1)
    x = tf.debugging.check_numerics(x, "NaN after Backbone")

    embds = OutputLayer(embd_shape, w_decay=w_decay)(x)

    if training:
        assert num_classes is not None
        labels = Input([], name='label')
        if head_type == 'ArcHead':
            logist = ArcHead(num_classes=num_classes, margin=margin,
                             logist_scale=logist_scale)(embds, labels)
        else:
            logist = NormHead(num_classes=num_classes, w_decay=w_decay)(embds)
        return Model((inputs, labels), logist, name=name)
    else:
        return Model(inputs, embds, name=name)
