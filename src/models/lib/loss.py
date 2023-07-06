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

    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(
            alpha * K.pow(1.0 - pt_1, gamma) * K.log(pt_1 + K.epsilon())
        ) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1.0 - pt_0 + K.epsilon()))

    return focal_loss_fixed
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

    # return tf.keras.losses.BinaryFocalCrossentropy(alpha=alpha, gamma=gamma)


def dice_coef(y_true, y_pred):
    smooth = K.epsilon()
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2.0 * intersection + smooth) / (
        K.sum(K.square(y_true_f)) + K.sum(K.square(y_pred_f)) + smooth
    )
    return dice


def dice_coef_nosq(y_true, y_pred):
    smooth = K.epsilon()
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice


def dice_loss(y_true, y_pred):
    dice = dice_coef(y_true, y_pred)
    loss = 1 - dice
    return loss


def dice_loss_nosq(y_true, y_pred):
    dice = dice_coef_nosq(y_true, y_pred)
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


def log_cosh_dice_loss_nosq(y_true, y_pred):
    dice = dice_loss_nosq(y_true, y_pred)
    return tf.math.log((tf.exp(dice) + tf.exp(-dice)) / 2.0)


def log_cosh_dice_focal(alpha=0.25, gamma=2.0):
    focal_func = categorical_focal_loss(alpha=alpha, gamma=gamma)

    def loss(y_true, y_pred):
        dice = log_cosh_dice_loss(y_true, y_pred)
        focal_loss = focal_func(y_true, y_pred)
        return dice + focal_loss

    return loss


def dyn_weighted_bincrossentropy(true, pred):
    """
    Calculates weighted binary cross entropy. The weights are determined dynamically
    by the balance of each category. This weight is calculated for each batch.

    The weights are calculted by determining the number of 'pos' and 'neg' classes
    in the true labels, then dividing by the number of total predictions.

    For example if there is 1 pos class, and 99 neg class, then the weights are 1/100 and 99/100.
    These weights can be applied so false negatives are weighted 99/100, while false postives are weighted
    1/100. This prevents the classifier from labeling everything negative and getting 99% accuracy.

    This can be useful for unbalanced catagories.

    """
    # get the total number of inputs
    num_pred = K.sum(K.cast(pred < 0.5, true.dtype)) + K.sum(true)

    # get weight of values in 'pos' category
    zero_weight = K.sum(true) / num_pred + K.epsilon()

    # get weight of values in 'false' category
    one_weight = K.sum(K.cast(pred < 0.5, true.dtype)) / num_pred + K.epsilon()

    # calculate the weight vector
    weights = (1.0 - true) * zero_weight + true * one_weight

    # calculate the binary cross entropy
    bin_crossentropy = K.binary_crossentropy(true, pred)

    # apply the weights
    weighted_bin_crossentropy = weights * bin_crossentropy

    return K.mean(weighted_bin_crossentropy)


def dice_coef_nosq(y_true, y_pred):
    smooth = K.epsilon()
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2.0 * intersection + smooth) / (
        K.sum(K.square(y_true_f)) + K.sum(K.square(y_pred_f)) + smooth
    )
    return dice
