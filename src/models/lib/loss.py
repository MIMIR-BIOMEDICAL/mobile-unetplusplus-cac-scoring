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

    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Compute mean loss in mini_batch
        return K.mean(K.sum(loss, axis=-1))

    return categorical_focal_loss_fixed


def dice_coef_func(use_bg=True):
    if use_bg:
        last_axis = 0
    else:
        last_axis = 1

    def dice_coef(y_true, y_pred):
        smooth = K.epsilon()
        y_true_f = K.flatten(y_true[:, :, :, last_axis:])
        y_pred_f = K.flatten(y_pred[:, :, :, last_axis:])
        intersect = K.sum(y_true_f * y_pred_f, axis=-1)
        denom = K.sum(y_true_f + y_pred_f, axis=-1)
        return K.mean((2.0 * intersect / (denom + smooth)))

    return dice_coef


def dice_loss_func(y_true, y_pred):
    dice = dice_coef_func()
    loss = 1 - dice(y_true, y_pred)
    return loss


def dice_focal(alpha=0.25, gamma=2.0):
    focal_func = categorical_focal_loss(alpha=alpha, gamma=gamma)

    def loss(y_true, y_pred):
        dice_loss = dice_loss_func(y_true, y_pred)
        focal_loss = focal_func(y_true, y_pred)
        return dice_loss + focal_loss

    return losss


def log_cosh_dice_loss(y_true, y_pred):
    dice_loss = dice_loss_func(y_true, y_pred)
    return tf.math.log((tf.exp(dice_loss) + tf.exp(-dice_loss)) / 2.0)


def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss
