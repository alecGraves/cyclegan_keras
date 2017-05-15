'''
Loss functions for CycleGan
'''
import keras.backend as k
from keras.losses import mean_squared_error
_discriminator_train_thresh = 0.2

def descriminator_loss(y_true, y_pred):
    loss = mean_squared_error(y_true, y_pred)
    is_large = k.greater(loss, _discriminator_train_thresh) # threshold
    return loss * is_large

def cycle_loss(y_true, y_pred):
    if k.image_data_format() is 'channels_first':
        x_w, x_h = (2, 3)
    else:
        x_w, x_h = (1, 2)
    loss = k.abs(k.max((y_true - y_pred, k.epsilon)))
    loss = k.sum(loss, x_h)
    loss = k.sum(loss, x_w)
    return loss

