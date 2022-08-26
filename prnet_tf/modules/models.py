import functools
import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.keras import Model
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Flatten,
    Input,
    Conv2D,
    Conv2DTranspose,
    ReLU
)


def _regularizer(weights_decay=5e-4):
    return tf.keras.regularizers.l2(weights_decay)


def _kernel_init(scale=1.0, seed=None):
    """He normal initializer with scale."""
    scale = 2. * scale
    return tf.keras.initializers.VarianceScaling(
        scale=scale, mode='fan_in', distribution="truncated_normal", seed=seed)


class BatchNormalization(tf.keras.layers.BatchNormalization):
    """Make trainable=False freeze BN for real (the og version is sad).
       ref: https://github.com/zzh8829/yolov3-tf2
    """
    # sbasak01 to do : Check if the mometum = 0.9 eps=1e-5 will be greater?
    # old: momentum=0.99, epsilon=1e-3
    def __init__(self, axis=-1, momentum=0.9, epsilon=1e-5, center=True,
                scale=True, name=None, **kwargs):
        super(BatchNormalization, self).__init__(
            axis=axis, momentum=momentum, epsilon=epsilon, center=center,
            scale=scale, name=name, **kwargs)

    def call(self, x, training=False):
        if training is None:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


class ResBlock(tf.keras.layers.Layer):
    """Residual Dense Block"""
    def __init__(self, num_outputs, kernel_size=4, strides=1, wd=0.,
                 name='resblock', **kwargs):
        super(ResBlock, self).__init__(name=name, **kwargs)
        self.strides = strides
        self.num_outputs = num_outputs
        _Conv2DLayer = functools.partial(Conv2D,
                padding='same',
                kernel_initializer=_kernel_init(0.1), bias_initializer='zeros',
                kernel_regularizer=_regularizer(wd))
        self.conv_sc = _Conv2DLayer(
                filters=num_outputs, kernel_size=1, strides=strides)
        self.conv1 = _Conv2DLayer(
                filters=num_outputs // 2, kernel_size=1, strides=1,
                activation=ReLU())
        self.conv2 = _Conv2DLayer(
                filters=num_outputs // 2, kernel_size=kernel_size, strides=strides,
                activation=ReLU())
        self.conv3 = _Conv2DLayer(
                filters=num_outputs, kernel_size=1, strides=1)
        self.b1 = BatchNormalization()
        self.b2 = BatchNormalization()
        self.b3 = BatchNormalization()

    def call(self, x):
        shortcut = x
        if self.strides != 1 or x.get_shape()[3] != self.num_outputs:
            shortcut = self.conv_sc(shortcut)
        x = self.conv1(x)
        x = self.b1(x)
        x = self.conv2(x)
        x = self.b2(x)
        x = self.conv3(x)
        x += shortcut
        x = self.b3(x)
        x = ReLU()(x)
        return x


class DeconvUnit(tf.keras.layers.Layer):
    """Residual Dense Block"""
    def __init__(self, num_outputs, strides=1, wd=0., activation=ReLU(),
                 name='conv_trans_unit', is_bn=False, **kwargs):
        super(DeconvUnit, self).__init__(name=name, **kwargs)
        self.convtrans = Conv2DTranspose(filters=num_outputs,
                                         strides=strides,
                                         kernel_size=4,
                                         padding='same',
                                         bias_initializer='zeros',
                                         activation=activation,
                                         kernel_initializer=_kernel_init(),
                                         kernel_regularizer=_regularizer(wd))
        self.is_bn = is_bn
        self.bn = BatchNormalization()

    def call(self, x):
        x = self.convtrans(x)
        if self.is_bn:
            x = self.bn(x)
        return x


def PRNet_Model(in_size, channels, wd=0., name='PRNet_Model'):
    """Residual-in-Residual Dense Block based Model """
    conv_f = functools.partial(Conv2D, kernel_size=4, padding='same',
                               bias_initializer='zeros',
                               kernel_initializer=_kernel_init(),
                               kernel_regularizer=_regularizer(wd))

    # encoder
    size = 16
    x = inputs = Input([in_size, in_size, channels], name='input_image')
    x = conv_f(filters=size, strides=1, activation=ReLU())(x)  # 256 x 256 x 16
    x = ResBlock(size * 2, strides=2, name='resblock_1')(x)  # 128 x 128 x 32
    x = ResBlock(size * 2, strides=1, name='resblock_2')(x)  # 128 x 128 x 32
    x = ResBlock(size * 4, strides=2, name='resblock_3')(x)  # 64 x 64 x 64
    x = ResBlock(size * 4, strides=1, name='resblock_4')(x)  # 64 x 64 x 64
    x = ResBlock(size * 8, strides=2, name='resblock_5')(x)  # 32 X 32 X 128
    x = ResBlock(size * 8, strides=1, name='resblock_6')(x)  # 32 x 32 x 128
    x = ResBlock(size * 16, strides=2, name='resblock_7')(x)  # 16 x 16 x 256
    x = ResBlock(size * 16, strides=1, name='resblock_8')(x)  # 16 x 16 x 256
    x = ResBlock(size * 32, strides=2, name='resblock_9')(x)  # 8 x 8 x 512
    x = ResBlock(size * 32, strides=1, name='resblock_10')(x)  # 8 x 8 x 512

    # decoder
    x = DeconvUnit(size * 32, strides=1, name='deconv_1')(x)  # 8 x 8 x 512
    x = DeconvUnit(size * 16, strides=2, name='deconv_2')(x)  # 16 x 16 x 256
    x = DeconvUnit(size * 16, strides=1, name='deconv_3')(x)  # 16 x 16 x 256
    x = DeconvUnit(size * 16, strides=1, name='deconv_4')(x)  # 16 x 16 x 256
    x = DeconvUnit(size * 8, strides=2, name='deconv_5')(x)  # 32 x 32 x 128
    x = DeconvUnit(size * 8, strides=1, name='deconv_6')(x)  # 32 x 32 x 128
    x = DeconvUnit(size * 8, strides=1, name='deconv_7')(x)  # 32 x 32 x 128
    x = DeconvUnit(size * 4, strides=2, name='deconv_8')(x)  # 64 x 64 x 64
    x = DeconvUnit(size * 4, strides=1, name='deconv_9')(x)  # 64 x 64 x 64
    x = DeconvUnit(size * 4, strides=1, name='deconv_10')(x)  # 64 x 64 x 64
    x = DeconvUnit(size * 2, strides=2, name='deconv_11')(x)  # 128 x 128 x 32
    x = DeconvUnit(size * 2, strides=1, name='deconv_12')(x)  # 128 x 128 x 32
    x = DeconvUnit(size, strides=2, name='deconv_13')(x)  # 256 x 256 x 16
    x = DeconvUnit(size, strides=1, name='deconv_14')(x)  # 256 x 256 x 16
    x = DeconvUnit(3, strides=1, name='deconv_15')(x)  # 256 x 256 x 3
    x = DeconvUnit(3, strides=1, name='deconv_16')(x)  # 256 x 256 x 3
    out = DeconvUnit(3, strides=1, activation=sigmoid, name='deconv_out')(x)  # 256 x 256 x 3

    return Model(inputs, out, name=name)


class PosPrediction():
    def __init__(self, cfg, resolution_inp = 256, resolution_op = 256):
        # -- hyper settings
        self.resolution_inp = resolution_inp
        self.resolution_op = resolution_op
        self.cfg = cfg
        # network type
        self.network = PRNet_Model(cfg['input_size'], cfg['ch_size'])

    def restore(self):
        # load checkpoint
        checkpoint_dir = './checkpoints/' + self.cfg['sub_name']
        checkpoint = tf.train.Checkpoint(model=self.network)
        if tf.train.latest_checkpoint(checkpoint_dir):
            checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
            print("[*] load ckpt from {}.".format(
                tf.train.latest_checkpoint(checkpoint_dir)))
        else:
            print("[*] Cannot find ckptfrom {}.".format(
                tf.train.latest_checkpoint(checkpoint_dir)))

    def predict(self, image):
        pos = self.network(image[np.newaxis, ::])
        pos = np.squeeze(pos.numpy())
        return pos

    def predict_batch(self, images):
        pos = self.network(images)
        pos = pos.numpy()
        return pos
