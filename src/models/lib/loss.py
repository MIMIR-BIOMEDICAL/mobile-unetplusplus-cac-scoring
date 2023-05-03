"""Module for log cosh loss"""
import keras.backend as K
import tensorflow as tf


def log_cosh_loss(y_true, y_pred):
    """
    An implementation of log cosh loss based on
    'A survey of loss functions for semantic segmentation'
    by Shruti Jadon

    Loss implementation based on
    https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions
    Survey Paper DOI: 10.1109/CIBCB48159.2020.9277638

    Args:
        y_true ():
        y_pred ():

    Returns:

    """
    smooth = 1.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    dice_loss = 1 - score
    log_cosh = tf.math.log((tf.exp(dice_loss) + tf.exp(-dice_loss)) / 2.0)
    return log_cosh
