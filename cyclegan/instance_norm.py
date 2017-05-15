'''
This is a keras implementation of the Instance Normalization layer.
Developed with help from:
 - BatchNorm paper: 
     - https://arxiv.org/abs/1502.03167
 - Batchnorm Keras:
     - https://github.com/fchollet/keras/blob/master/keras/layers/normalization.py
 - InstanceNormalization paper:
     - https://arxiv.org/abs/1607.08022
 - Keras-Theano implementation:
     - https://github.com/jayanthkoushik/neural-style/blob/master/neural_style/fast_neural_style/transformer_net.py
'''
from keras.engine.topology import Layer
import keras.backend as k

class InstanceNormalization(Layer):
    def __init__(self,
                 epsilon=1e-3,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 **kwargs):
        super().__init__(**kwargs)
        if k.image_data_format() is 'channels_first':
            self.axis = 1
        else: # image channels x.shape[3]
            self.axis = 3
        self.epsilon = epsilon
        self.beta_initializer = beta_initializer
        self.gamma_initializer = gamma_initializer

    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(input_shape[self.axis],),
                                     initializer=self.gamma_initializer,
                                     trainable=True)
        self.beta = self.add_weight(shape=(input_shape[self.axis],),
                                    initializer=self.beta_initializer,
                                    trainable=True)
        super().build(input_shape)

    def call(self, x, mask=None):
        # image channel indices
        x_w, x_h = [1, 2, 3].remove(self.axis) # spatial dims of input
        hw = k.cast(x.shape[x_h]* x.shape[x_w], k.floatx())
        mu = x.sum(axis=x_w).sum(axis=x_h) / hw # instance means
        mu = mu.dimshuffle(0, 1, 'x', 'x') # reshape for varience calculation and norm
        sig_sq = k.square(x - mu).sum(axis=x_w).sum(axis=x_h) # instance variences
        x_hat = (x - mu) / k.sqrt(sig_sq.dimshuffle(0, 1, 'x', 'x') +
                                    self.epsilon) # normalize
        return (self.gamma.dimshuffle('x', 0, 'x', 'x') * x_hat +
            self.beta.dimshuffle('x', 0, 'x', 'x')) # scale and shift

