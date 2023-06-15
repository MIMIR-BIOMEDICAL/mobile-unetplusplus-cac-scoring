"""Module for unified focal loss from https://github.com/mlyg/unified-focal-loss"""
import tensorflow as tf
from tensorflow.keras import backend as K


def identify_axis(shape):
    # Three dimensional
    if len(shape) == 5:
        return [1, 2, 3]
    # Two dimensional
    elif len(shape) == 4:
        return [1, 2]
    # Exception - Unknown
    else:
        raise ValueError("Metric: Shape of tensor is neither 2D or 3D.")


################################
#     Asymmetric Focal loss    #
################################
def asymmetric_focal_loss(delta=0.7, gamma=2.0):
    """For Imbalanced datasets
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        Focal Tversky loss' focal parameter controls degree of down-weighting of easy examples, by default 2.0
    """

    def loss_function(y_true, y_pred):
        axis = identify_axis(y_true.get_shape())

        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * K.log(y_pred)

        # calculate losses separately for each class, only suppressing background class
        back_ce = K.pow(1 - y_pred[:, :, :, 0], gamma) * cross_entropy[:, :, :, 0]
        back_ce = (1 - delta) * back_ce

        fore_ce = cross_entropy[:, :, :, 1]
        fore_ce = delta * fore_ce

        loss = K.mean(K.sum(tf.stack([back_ce, fore_ce], axis=-1), axis=-1))

        return loss

    return loss_function


#################################
# Asymmetric Focal Tversky loss #
#################################
def asymmetric_focal_tversky_loss(delta=0.7, gamma=0.75):
    """This is the implementation for binary segmentation.
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 0.75
    """

    def loss_function(y_true, y_pred):
        # Clip values to prevent division by zero error
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)

        axis = identify_axis(y_true.get_shape())
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)
        tp = K.sum(y_true * y_pred, axis=axis)
        fn = K.sum(y_true * (1 - y_pred), axis=axis)
        fp = K.sum((1 - y_true) * y_pred, axis=axis)
        dice_class = (tp + epsilon) / (tp + delta * fn + (1 - delta) * fp + epsilon)

        # calculate losses separately for each class, only enhancing foreground class
        back_dice = 1 - dice_class[:, 0]
        fore_dice = (1 - dice_class[:, 1]) * K.pow(1 - dice_class[:, 1], -gamma)

        # Average class scores
        loss = K.mean(tf.stack([back_dice, fore_dice], axis=-1))
        return loss

    return loss_function


###########################################
#      Asymmetric Unified Focal loss      #
###########################################
def asym_unified_focal_loss(weight=0.5, delta=0.6, gamma=0.5):
    """The Unified Focal loss is a new compound loss function that unifies Dice-based and cross entropy-based loss functions into a single framework.
    Parameters
    ----------
    weight : float, optional
        represents lambda parameter and controls weight given to asymmetric Focal Tversky loss and asymmetric Focal loss, by default 0.5
    delta : float, optional
        controls weight given to each class, by default 0.6
    gamma : float, optional
        focal parameter controls the degree of background suppression and foreground enhancement, by default 0.5
    """

    def loss_function(y_true, y_pred):
        asymmetric_ftl = asymmetric_focal_tversky_loss(delta=delta, gamma=gamma)(
            y_true, y_pred
        )
        asymmetric_fl = asymmetric_focal_loss(delta=delta, gamma=gamma)(y_true, y_pred)
        if weight is not None:
            return (weight * asymmetric_ftl) + ((1 - weight) * asymmetric_fl)
        else:
            return asymmetric_ftl + asymmetric_fl

    return loss_function


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
        y_true = tf.cast(y_true, tf.float32)
        # Define epsilon so that the back-propagation will not result in NaN for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        # y_pred = y_pred + epsilon
        # Clip the prediciton value
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        # Calculate p_t
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        # Calculate alpha_t
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        # Calculate cross entropy
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.mean(K.sum(loss, axis=1))
        return loss

    return categorical_focal_loss_fixed


def dice_coef(y_true, y_pred):
    axis = identify_axis(y_true.get_shape())
    # Calculate true positives (tp), false negatives (fn) and false positives (fp)
    tp = K.sum(y_true * y_pred, axis=axis)
    fn = K.sum(y_true * (1 - y_pred), axis=axis)
    fp = K.sum((1 - y_true) * y_pred, axis=axis)
    # Calculate Dice score
    dice_class = (tp + smooth) / (tp + delta * fn + (1 - delta) * fp + smooth)
    return dice_class


def dice_loss(y_true, y_pred):
    dice = dice_coef(y_true, y_pred)
    loss = 1 - dice
    return loss


def dice_coef_no_bg(y_true, y_pred):
    smooth = K.epsilon()
    y_true_f = K.flatten(y_true[:, :, :, 1:])
    y_pred_f = K.flatten(y_pred[:, :, :, 1:])
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
    return K.mean((2.0 * intersect / (denom + smooth)))


def dice_loss_no_bg(y_true, y_pred):
    dice = dice_coef_no_bg(y_true, y_pred)
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


def log_cosh_dice_loss_no_bg(y_true, y_pred):
    dice = dice_loss_no_bg(y_true, y_pred)
    return tf.math.log((tf.exp(dice) + tf.exp(-dice)) / 2.0)


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
