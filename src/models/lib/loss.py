"""Module for unified focal loss from https://github.com/mlyg/unified-focal-loss"""
import tensorflow as tf
from tensorflow.keras import backend as K


def categorical_focal_loss(alpha=0.25, gamma=2.0):
    """
    https://github.com/umbertogriffo/focal-loss-keras

    Softmax version of focal loss.
    When there is a skew between different categories/labels in your data set, you can try to apply this function as a
    loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1

      where m = number of classes, c = class and o = observation

    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy. Alpha is used to specify the weight of different
      categories/labels, the size of the array needs to be consistent with the number of classes.
      gamma -- focusing parameter for modulating factor (1-p)

    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper

    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy

    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    # def categorical_focal_loss_fixed(y_true, y_pred):
    # """
    # :param y_true: A tensor of the same shape as `y_pred`
    # :param y_pred: A tensor resulting from a softmax
    # :return: Output tensor.
    # """
    # y_true = tf.cast(y_true, tf.float32)
    # # Define epsilon so that the back-propagation will not result in NaN for 0 divisor case
    # epsilon = K.epsilon()
    # # Add the epsilon to prediction value
    # # y_pred = y_pred + epsilon
    # # Clip the prediciton value
    # y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    # # Calculate p_t
    # p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
    # # Calculate alpha_t
    # alpha_factor = K.ones_like(y_true) * alpha
    # alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
    # # Calculate cross entropy
    # cross_entropy = -K.log(p_t)
    # weight = alpha_t * K.pow((1 - p_t), gamma)
    # # Calculate focal loss
    # loss = weight * cross_entropy
    # # Sum the losses in mini_batch
    # loss = K.mean(K.sum(loss, axis=-1))
    # return loss

    # return categorical_focal_loss_fixed
    return tf.keras.losses.BinaryFocalCrossentropy(alpha=alpha, gamma=gamma)


def dice_coef(y_true, y_pred):
    smooth = 1
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    dice = K.mean((2.0 * intersection + smooth) / (union + smooth), axis=0)
    return dice


def dice_loss(y_true, y_pred):
    dice = dice_coef(y_true, y_pred)
    loss = 1 - dice
    return loss


def dice_focal(alpha=0.25, gamma=2.0):
    focal_func = categorical_focal_loss(alpha=alpha, gamma=gamma)

    def loss(y_true, y_pred):
        dice = dice_loss(y_true, y_pred)
        focal_loss = focal_func(y_true, y_pred)
        return dice + focal_loss

    return loss


def dice_focal_no_bg(alpha=0.25, gamma=2.0):
    focal_func = categorical_focal_loss(alpha=alpha, gamma=gamma)

    def loss(y_true, y_pred):
        dice = dice_loss_no_bg(y_true, y_pred)
        focal_loss = focal_func(y_true, y_pred)
        return dice + focal_loss

    return loss


def log_cosh_dice_loss(y_true, y_pred):
    dice = dice_loss(y_true, y_pred)
    return tf.math.log((tf.exp(dice) + tf.exp(-dice)) / 2.0)


def log_cosh_dice_focal(alpha=0.25, gamma=2.0):
    focal_func = categorical_focal_loss(alpha=alpha, gamma=gamma)

    def loss(y_true, y_pred):
        dice = log_cosh_dice_loss(y_true, y_pred)
        focal_loss = focal_func(y_true, y_pred)
        return dice + focal_loss

    return loss
