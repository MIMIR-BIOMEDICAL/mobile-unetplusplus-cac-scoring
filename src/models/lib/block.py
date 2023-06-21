"""Module containing the building block function for the ml model"""
from typing import Callable

import keras
import tensorflow as tf
from tensorflow.keras import layers


# original block
def conv_bn_relu_unit(node_name, n_filter, batch_norm, n_kernel=3) -> Callable:
    """
    A block containing convolution layer with batch normalization and ReLU activation

    Args:
        node_name (str): Name of the node/block
        n_filter (int): Number of filters in the convolution layer
        batch_norm (bool): Flag to enable/disable batch normalization
        n_kernel (int): Size of the convolution kernel

    Returns:
        Callable: A callable function that accepts an input tensor and applies the convolution,
                  batch normalization and activation to it.
    """

    def layer(input_tensor):
        x = layers.Conv2D(
            filters=n_filter,
            kernel_size=n_kernel,
            name=f"x_{node_name}_conv",
            padding="same",
        )(input_tensor)

        # Batch Normalization
        if batch_norm:
            x = layers.BatchNormalization(name=f"x_{node_name}_bn")(x)
            # Activation
            x = layers.Activation("relu", name=f"x_{node_name}_activation")(x)
            x = layers.Dropout(0.2, name=f"x_{node_name}_drop")(x)
        else:
            # Activation
            x = layers.Activation("relu", name=f"x_{node_name}_activation")(x)

        return x

    return layer


def conv_bn_relu_block(
    node_name,
    n_filter,
    batch_norm,
    n_kernel=3,
    mode="upsample",
) -> Callable:
    """
    Creates a block of convolution, batch normalization, and ReLU activation base on upsample mode.

    Args:
        node_name (str): Name of the node.
        n_filter (int): Number of convolutional filters.
        batch_norm (bool): Whether to use batch normalization.
        n_kernel (int): Size of the convolution kernel. Default is 3.
        mode (str): Type of upsampling to use. Can be either "upsample" or "transpose". Default is "upsample".

    Returns:
        A Keras layer that applies a block of convolution, batch normalization, and ReLU activation.
    """

    if mode == "upsample":

        def layer(input_tensor):
            x = conv_bn_relu_unit(
                node_name=node_name,
                n_filter=n_filter,
                batch_norm=batch_norm,
                n_kernel=n_kernel,
            )(input_tensor)
            return x

    elif mode == "transpose":

        def layer(input_tensor):
            x = conv_bn_relu_unit(
                node_name=f"{node_name}_1",
                n_filter=n_filter,
                batch_norm=batch_norm,
                n_kernel=n_kernel,
            )(input_tensor)
            x = conv_bn_relu_unit(
                node_name=f"{node_name}_2",
                n_filter=n_filter,
                batch_norm=batch_norm,
                n_kernel=n_kernel,
            )(x)

            return x

    else:
        raise ValueError("Mode can only be either upsample or transpose")

    return layer


def upsample_block(
    node_name, n_filter, batch_norm, n_kernel=2, mode="upsample"
) -> Callable:
    """
    Upsample block containing different types of upsampling methods.

    Args:
        node_name (str): Name of the layer
        n_filter (int): Number of filters in the convolution layer
        batch_norm (bool): Whether to use batch normalization or not
        n_kernel (int): Kernel size of the convolution layer
        mode (str): Upsampling mode. Can be either "upsample" or "transpose".

    Returns:
        A Keras layer that performs upsampling.
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

            # Batch Normalization
            if batch_norm:
                x = layers.BatchNormalization(name=f"x_{node_name}_transpose_bn")(x)
                # Activation
                x = layers.Activation(
                    "relu", name=f"x_{node_name}_transpose_activation"
                )(x)
                x = layers.Dropout(0.2, name=f"x_{node_name}_transpose_drop")(x)
            else:
                # Activation
                x = layers.Activation("relu", name=f"x_{node_name}_activation")(x)

            return x

    else:
        raise ValueError("Mode can only be either upsample or transpose")

    return layer


###
# MOBILENETV2 Blocks
###


def pointwise_block(
    node_name: str,
    n_filter: int,
    batch_norm: bool,
    linear: bool,
    strides: int = 1,
    kernel: int = 1,
) -> Callable:
    """
    Returns a pointwise convolutional layer with optional batch normalization and activation.

    Args:
        node_name (str): Name of the layer.
        n_filter (int): Number of filters for the convolutional layer.
        batch_norm (bool): Whether to apply batch normalization.
        linear (bool): Whether to apply activation (relu6) after the convolutional layer.
        strides (int, optional): Stride of the convolutional layer. Defaults to 1.
        kernel (int, optional): Kernel size of the convolutional layer. Defaults to 1.

    Returns:
        callable: Function that returns the layer when called with an input tensor.
    """

    def layer(input_tensor: tf.Tensor) -> tf.Tensor:
        """
        A pointwise convolution block with added linearity or non-linearity (ReLU6).

        Args:
            input_tensor (tf.Tensor): Input tensor to the layer.

        Returns:
            tf.Tensor: Output tensor of the layer.
        """
        x = layers.Conv2D(
            n_filter, kernel, strides=strides, padding="same", name=f"{node_name}_pwise"
        )(input_tensor)
        if batch_norm:
            x = layers.BatchNormalization(name=f"{node_name}_pwise_bnorm")(x)
        if not linear:
            x = layers.Activation(tf.nn.relu6, name=f"{node_name}_pwise_relu6")(x)
        if batch_norm:
            x = layers.Dropout(0.2, name=f"{node_name}_pwise_drop")(x)
        return x

    return layer


def depthwise_block(
    node_name: str, batch_norm: bool, strides: int = 1, kernel: int = 3
) -> Callable:
    """Create a depthwise convolution block with relu6 activation.

    Args:
        node_name (str): Name of the block.
        batch_norm (bool): Whether or not to apply batch normalization.
        strides (int, optional): The strides of the convolution along the height and width. Defaults to 1.
        kernel (int, optional): Integer, the size of the kernel to be used in depthwise convolution. Defaults to 3.

    Returns:
        Callable: A callable object that applies the depthwise convolution block to the input tensor.
    """

    def layer(input_tensor: tf.Tensor) -> tf.Tensor:
        """Applies depthwise convolution block to input_tensor.

        Args:
            input_tensor (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Output tensor.
        """
        x = layers.DepthwiseConv2D(
            kernel, strides=strides, padding="same", name=f"{node_name}_dwise"
        )(input_tensor)

        if batch_norm:
            x = layers.BatchNormalization(name=f"{node_name}_dwise_bnorm")(x)
            # Activation
            x = layers.Activation(tf.nn.relu6, name=f"{node_name}_dwise_relu6")(x)
            x = layers.Dropout(0.2, name=f"x_{node_name}_dwise_drop")(x)
        else:
            # Activation
            x = layers.Activation(tf.nn.relu6, name=f"{node_name}_dwise_relu6")(x)

        return x

    return layer


def inverted_residual_bottleneck_block(
    node_name: str,
    n_filter: int,
    strides: int,
    t_expansion: int,
    batch_norm: bool,
    residual: bool = False,
) -> Callable:
    """
    A bottleneck block containing expansion and compression using pointwise and depthwise convolutions
    followed by optional residual connection.

    Args:
        node_name: A name for the block.
        n_filter: Number of filters in the output tensor.
        strides: Stride size of the depthwise convolution.
        t_expansion: Expansion factor for the number of filters in the expansion layer.
        batch_norm: Whether to apply batch normalization after each convolution.
        residual: Whether to apply a residual connection to the output tensor.

    Returns:
        A callable that takes an input tensor and returns the output tensor.
    """

    def layer(input_tensor: tf.Tensor) -> tf.Tensor:
        expanded_filter = keras.backend.int_shape(input_tensor)[-1] * t_expansion

        # Expansion layer
        x = pointwise_block(
            node_name=node_name + "_expand",
            n_filter=expanded_filter,
            batch_norm=batch_norm,
            linear=False,
        )(input_tensor)

        # Depthwise Layer
        x = depthwise_block(
            node_name=node_name + "_depthwise",
            batch_norm=batch_norm,
            strides=strides,
            kernel=3,
        )(x)

        # Compression layer
        x = pointwise_block(
            node_name=node_name + "_compress",
            n_filter=n_filter,
            batch_norm=batch_norm,
            strides=1,
            kernel=1,
            linear=True,
        )(x)

        if residual:
            x = layers.Add(name=f"{node_name}_add")([x, input_tensor])

        return x

    return layer


def sequence_inv_res_bot_block(
    node_name, n_filter, batch_norm, strides, t_expansion, n_iter
) -> Callable:
    """
    A layer containing a sequence of inverted
    residual bottleneck block that is repeated
    n_iter times

    Args:
        node_name (str):
        n_filter (int):
        batch_norm (bool):
        strides (int):
        t_expansion (int):
        n_iter (int):

    Returns:
        Callable: A callable Keras layer that applies the sequence of inverted residual bottleneck blocks to an input tensor.
    """

    def layer(input_tensor):
        x = inverted_residual_bottleneck_block(
            node_name=f"x_{node_name}_iter0",
            n_filter=n_filter,
            batch_norm=batch_norm,
            strides=strides,
            t_expansion=t_expansion,
            residual=False,
        )(input_tensor)

        for index in range(1, n_iter):
            x = inverted_residual_bottleneck_block(
                node_name=f"x_{node_name}_iter{index}",
                n_filter=n_filter,
                batch_norm=batch_norm,
                strides=1,
                t_expansion=t_expansion,
                residual=True,
            )(x)
        return x

    return layer
