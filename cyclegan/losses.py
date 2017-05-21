'''
Loss functions for CycleGan
'''
import keras.backend as k
from keras.losses import mean_squared_error

_disc_train_thresh = 0.0

def discriminator_loss(y_true, y_pred):
    loss = mean_squared_error(y_true, y_pred)
    is_large = k.greater(loss, k.constant(_disc_train_thresh)) # threshold
    is_large = k.cast(is_large, k.floatx())
    return loss * is_large # binary threshold the loss to prevent overtraining the discriminator

def cycle_loss(y_true, y_pred):
    if k.image_data_format() is 'channels_first':
        x_w = 2
        x_h = 3
    else:
        x_w = 1
        x_h = 2
    loss = k.abs(y_true - y_pred)
    loss = k.sum(loss, axis=x_h)
    loss = k.sum(loss, axis=x_w)
    return loss

