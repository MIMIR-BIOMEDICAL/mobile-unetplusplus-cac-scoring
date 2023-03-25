"""Module containing the building block function for the ml model"""
import pathlib
import sys

import keras
import tensorflow as tf
from tensorflow.keras import layers

sys.path.append(pathlib.Path.cwd().as_posix())


# original block
def conv_relu_unit(node_name, n_filter, enable_batch_norm, dropout_rate=1, n_kernel=3):
    """
    Smallest unit  containing convolution with batch norm and dropout

    Args:
        node_name ():
        n_filter ():
        enable_batch_norm ():
        dropout_rate ():
        n_kernel ():

    Returns:

    """

    def layer(input_tensor):
        x = layers.Conv2D(
            filters=n_filter,
            kernel_size=n_kernel,
            name=f"x_{node_name}_conv",
            padding="same",
        )(input_tensor)

        # Batch Normalization
        if enable_batch_norm:
            x = layers.BatchNormalization(name=f"x_{node_name}_bn")(x)

        # ACtivation
        x = layers.Activation("relu", name=f"x_{node_name}_activation")(x)

        # Dropout
        if dropout_rate != 1:
            x = layers.Dropout(dropout_rate, name=f"x_{node_name}_dropout")(x)
        return x

    return layer


def conv_relu_block(
    node_name, n_filter, enable_batch_norm, dropout_rate=1, n_kernel=3, mode="upsample"
):
    """
    The activation block for different type of upsampling

    Args:
        node_name ():
        n_filter ():
        enable_batch_norm ():
        dropout_rate ():
        n_kernel ():
        mode ():

    Returns:

    """
    if mode == "upsample":

        def layer(input_tensor):
            x = conv_relu_unit(
                node_name, n_filter, enable_batch_norm, dropout_rate, n_kernel
            )(input_tensor)
            return x

    elif mode == "transpose":

        def layer(input_tensor):
            x = conv_relu_unit(
                node_name + "_1", n_filter, enable_batch_norm, dropout_rate, n_kernel
            )(input_tensor)
            x = conv_relu_unit(
                node_name + "_2", n_filter, enable_batch_norm, dropout_rate, n_kernel
            )(x)
            return x

    else:
        raise ValueError("Mode can only be either upsample or transpose")

    return layer


def upsample_layer(
    node_name, n_filter, enable_batch_norm, dropout_rate=1, n_kernel=2, mode="upsample"
):
    """
    Upsample block containing different type of upsampling method

    Args:
        node_name ():
        n_filter ():
        enable_batch_norm ():
        dropout_rate ():
        n_kernel ():
        mode ():

    Returns:

    """
    if mode == "upsample":

        def layer(input_tensor):
            x = layers.UpSampling2D(size=2, name=f"x_{node_name}_upsample")(
                input_tensor
            )
            return x

    elif mode == "transpose":

        def layer(input_tensor):
            x = layers.Conv2DTranspose(
                filters=n_filter,
                kernel_size=n_kernel,
                strides=2,
                name=f"x_{node_name}_transpose",
                padding="same",
            )(input_tensor)

            # Batch Norm
            if enable_batch_norm:
                x = layers.BatchNormalization(name=f"x_{node_name}_transpose_bn")(x)

            # ACtivation
            x = layers.Activation("relu", name=f"x_{node_name}_transpose_activation")(x)

            # Dropout
            if dropout_rate != 1:
                x = layers.Dropout(
                    dropout_rate, name=f"x_{node_name}_transpose_dropout"
                )(x)
            return x

    else:
        raise ValueError("Mode can only be either upsample or transpose")

    return layer


###
# MOBILENETV2 Blocks
###


def pointwise_block(node_name, filters, linear: bool, strides=1, kernel=1):
    """
    A pointwise convolution block with added linearity
    or non linearity (relu6)

    Args:
        node_name ():
        filters ():
        linear ():
        strides ():
        kernel ():

    Returns:

    """

    def layer(input_tensor):
        x = layers.Conv2D(
            filters, kernel, strides=strides, padding="same", name=f"{node_name}_pwise"
        )(input_tensor)
        x = layers.BatchNormalization(name=f"{node_name}_pwise_bnorm")(x)
        if not linear:
            x = layers.Activation(tf.nn.relu6, name=f"{node_name}_pwwise_relu6")(x)
        return x

    return layer


def depthwise_block(node_name, strides=1, kernel=3):
    """
    A depthwise convolution block with relu6 activation

    Args:
        strides ():
        kernel ():

    Returns:

    """

    def layer(input_tensor):
        x = layers.DepthwiseConv2D(
            kernel, strides=strides, padding="same", name=f"{node_name}_dwise"
        )(input_tensor)
        x = layers.BatchNormalization(name=f"{node_name}_dwise_bnorm")(x)
        x = layers.Activation(tf.nn.relu6, name=f"{node_name}_dwise_relu6")(x)
        return x

    return layer


def inverted_residual_bottleneck_block(
    node_name, filters, strides, t_expansion, residual=False
):
    """
    A bottleneck block containig expansion and compression using

    Args:
        node_name ():
        filters ():
        strides ():
        t_expansion ():
        residual ():

    Returns:

    """

    def layer(input_tensor):
        expanded_filter = keras.backend.int_shape(input_tensor)[-1] * t_expansion
        x = pointwise_block(node_name + "_expand", expanded_filter, linear=False)(
            input_tensor
        )
        x = depthwise_block(node_name, strides=strides)(x)
        x = pointwise_block(node_name + "_compress", filters, linear=True)(x)
        if residual:
            x = layers.Add(name=f"{node_name}_add")([x, input_tensor])

        return x

    return layer


def sequence_inv_res_bot_block(node_name, filters, strides, t_expansion, n):
    """
    A layer containing a sequence of inverted
    residual bottleneck block that is repeated
    n times

    Args:
        node_name ():
        filters ():
        strides ():
        t_expansion ():
        n ():

    Returns:

    """

    def layer(input_tensor):
        x = inverted_residual_bottleneck_block(
            node_name=f"{node_name}_iter0",
            filters=filters,
            strides=strides,
            t_expansion=t_expansion,
            residual=False,
        )(input_tensor)

        for index in range(1, n):
            x = inverted_residual_bottleneck_block(
                node_name=f"{node_name}_iter{index}",
                filters=filters,
                strides=1,
                t_expansion=t_expansion,
                residual=True,
            )(x)
        return x

    return layer
