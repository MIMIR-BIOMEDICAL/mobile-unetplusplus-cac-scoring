"""Module for unified focal loss from https://github.com/mlyg/unified-focal-loss"""
import tensorflow as tf
from tensorflow.keras import backend as K


def categorical_focal_loss(alpha=0.25, gamma=2.0):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(
            alpha * K.pow(1.0 - pt_1, gamma) * K.log(pt_1 + K.epsilon())
        ) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1.0 - pt_0 + K.epsilon()))

    return focal_loss_fixed


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
