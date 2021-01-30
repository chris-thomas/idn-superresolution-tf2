from tensorflow.python.keras.layers import Add, Conv2D, Input, Lambda, Dropout, GaussianDropout
from tensorflow.python.keras.models import Model
import tensorflow as tf
from model.common import normalize, denormalize, pixel_shuffle


def idn(scale, num_filters=64):
    initializer = tf.keras.initializers.he_uniform()
    x_in = Input(shape=(None, None, 3))
   
    t_image = Lambda(normalize)(x_in)
    #stddev=0.01
    #t_image = tf.keras.layers.GaussianNoise(stddev)(t_image)
    sh = tf.shape(t_image)
    newShape = 2 * sh[1:3]
    t_image_bicubic = tf.image.resize(t_image, newShape)

    # feature extraction block
    conv1 = Conv2D(num_filters, 3, padding='same', activation='swish', name='conv1', use_bias=False, kernel_initializer=initializer)(t_image)
    conv2 = Conv2D(num_filters, 3, padding='same', activation='swish', name='conv2', use_bias=False, kernel_initializer=initializer)(conv1)
    distillation = conv2
    
    # stacked information distillation blocks
    for i in range(8):
        name='distill/%i' % i
        distillationConv1 = Conv2D(48, 3, padding='same', activation='swish', name=name+'/conv1', use_bias=False, kernel_initializer=initializer)(distillation)
        n_filter=32
        filter_size=(3, 3)
        strides=(1, 1)
        n_group=4
        act=lrelu
        channels = int(distillationConv1.get_shape()[-1])

        if n_group == 1:
            groupConvOutputs = Conv2D(n_filter, filter_size, padding='same', use_bias=True, kernel_initializer=initializer)(distillationConv1)
        else:
            inputGroups = tf.split(axis=3, num_or_size_splits=n_group, value=distillationConv1)
            convGroups = [Conv2D(n_filter, filter_size, padding='same', use_bias=True, kernel_initializer=initializer)(i) for i in inputGroups]
            groupConvOutputs = tf.concat(axis=3, values=convGroups)

        if act:
            groupConvOutputs = lrelu(groupConvOutputs)

        tmp = Conv2D(64, 3, padding='same', activation='swish', name=name+'/conv3', use_bias=False, kernel_initializer=initializer)(groupConvOutputs)
        tmp1, tmp2 = tf.split(axis=3, num_or_size_splits=[16, 48], value=tmp)
        tmp2 = Conv2D(64, 3, padding='same', activation='swish', name=name+'/conv4', use_bias=False, kernel_initializer=initializer)(tmp2)

        n_filter=48
        filter_size=(3, 3)
        channels = int(tmp2.get_shape()[-1])

        if n_group == 1:
            groupConvOutputs = Conv2D(n_filter, filter_size, padding='same', use_bias=True, kernel_initializer=initializer)(tmp2)
        else:
            inputGroups = tf.split(axis=3, num_or_size_splits=n_group, value=tmp2)
            convGroups = [Conv2D(n_filter, filter_size, padding='same', use_bias=True, kernel_initializer=initializer)(i) for i in inputGroups]
            groupConvOutputs = tf.concat(axis=3, values=convGroups)

        if act:
            groupConvOutputs = lrelu(groupConvOutputs)

        tmp2 = Conv2D(80, 3, padding='same', activation='swish', name=name+'/conv6', use_bias=False, kernel_initializer=initializer)(groupConvOutputs)
        tmp3 = tf.concat(axis=3, values=[distillation, tmp1]) + tmp2
        distillation = Conv2D(64, 1, padding='same', activation='swish', name=name+'/conv7', use_bias=False, kernel_initializer=initializer)(tmp3)

    dropout = 0.5
    distillation = GaussianDropout(dropout)(distillation)
    
    output = upsample(distillation, scale=scale,features=64, name=str(scale)) + t_image_bicubic


    output = Lambda(denormalize)(output)
    return Model(x_in, output, name="idn")
    
def lrelu(x, alpha=0.05):
    return tf.maximum(alpha * x, x)
    
def _phase_shift(I, r):
    return Lambda(pixel_shuffle(scale=r))(I)

def PS(X, r, color=False):
    if color:
        Xc = tf.split(X, 3, 3)  # tf.split(value, num_or_size_splits, axis=0)
        X = tf.concat([_phase_shift(x, r) for x in Xc], 3)
    else:
        X = _phase_shift(X, r)
    return X

def upsample(x, scale=4, features=32, name=None):
    x = Conv2D(features, 3, padding='same')(x)
    ps_features = 3 * (scale ** 2)
    x = Conv2D(ps_features, 3, padding='same')(x)
    x = PS(x, scale, color=True)
    return x

