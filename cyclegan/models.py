'''
source for models
'''
from keras.layers import Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import Add
from keras.models import Model

from .instance_norm import InstanceNormalization

def conv_block(x0, scale):
    x = Conv2D(int(64*scale), (1, 1))(x0)
    x = InstanceNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(int(64*scale), (3, 3), padding='same')(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(int(256*scale), (1, 1))(x)
    x = InstanceNormalization()(x)

    x1 = Conv2D(int(256*scale), (1, 1))(x0)
    x1 = InstanceNormalization()(x1)

    x = Add()([x, x1])
    x = LeakyReLU()(x)
    return x

def identity_block(x0, scale):
    x = Conv2D(int(64*scale), (1, 1))(x0)
    x = InstanceNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(int(64*scale), (3, 3), padding='same')(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(int(256*scale), (1, 1))(x)
    x = InstanceNormalization()(x)

    x = Add()([x, x0])
    x = LeakyReLU()(x)
    return x

def residual_block(x0, scale=1, num_id=2):
    x0 = conv_block(x0, scale)
    for i in range(num_id):
        x0 = identity_block(x0, scale)
    return x0
