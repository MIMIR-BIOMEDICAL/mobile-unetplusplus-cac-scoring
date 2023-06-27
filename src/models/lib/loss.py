"""Module for unified focal loss from https://github.com/mlyg/unified-focal-loss"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


def categorical_focal_loss(alpha=0.25, gamma=2.0):
    def focal_loss_fixed(y_true, y_pred):
        # Clip values to prevent division by zero error
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * K.log(y_pred)

        if alpha is not None:
            alpha_weight = np.array(alpha, dtype=np.float32)
            focal_loss = alpha_weight * K.pow(1 - y_pred, gamma) * cross_entropy
        else:
            focal_loss = K.pow(1 - y_pred, gamma) * cross_entropy

        focal_loss = K.mean(K.sum(focal_loss, axis=[-1]))
        return focal_loss

    return focal_loss_fixed

def dice_coef_slice_nosq(y_true, y_pred):
    smooth = K.epsilon()
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2.0 * intersection + smooth) / (
        K.sum(y_true_f) + K.sum(y_pred_f) + smooth
    )
    return dice

def dice_coef_slice(y_true, y_pred):
    smooth = K.epsilon()
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2.0 * intersection + smooth) / (
        K.sum(K.square(y_true_f)) + K.sum(K.square(y_pred_f)) + smooth
    )
    return dice


def dice_coef(y_true, y_pred):
    smooth = K.epsilon()
    dice = 0
    for i in range(0, 5):
        y_true_bg = y_true[:, :, :, i]
        y_pred_bg = y_pred[:, :, :, i]
        dice += dice_coef_slice(y_true_bg, y_pred_bg)
    return dice / 5

def dice_coef_nosq(y_true, y_pred):
    smooth = K.epsilon()
    dice = 0
    for i in range(0, 5):
        y_true_bg = y_true[:, :, :, i]
        y_pred_bg = y_pred[:, :, :, i]
        dice += dice_coef_slice_nosq(y_true_bg, y_pred_bg)
    return dice / 5


def back_dice(y_true, y_pred):
    y_true_bg = y_true[:, :, :, 0]
    y_pred_bg = y_pred[:, :, :, 0]
    return dice_coef_slice(y_true_bg, y_pred_bg)


def fore_dice(y_true, y_pred):
    dice = 0
    for i in range(1, 5):
        y_true_bg = y_true[:, :, :, i]
        y_pred_bg = y_pred[:, :, :, i]
        dice += dice_coef_slice(y_true_bg, y_pred_bg)
    return dice / 4


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
