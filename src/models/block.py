"""Module containing the building block function for the ml model"""
import pathlib
import sys

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
