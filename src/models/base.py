"""Module containing the base ml model"""
import pathlib
import sys

from tensorflow import \
    keras  # pylint: disable=wrong-import-position,import-error
from tensorflow.keras import \
    layers  # pylint: disable=wrong-import-position,import-error

sys.path.append(pathlib.Path.cwd().as_posix())

from src.models.block import (  # pylint: disable=wrong-import-position,import-error
    conv_relu_block, sequence_inv_res_bot_block, upsample_layer)
from src.models.utils import \
    node_name_func  # pylint: disable=wrong-import-position,import-error


def base_unet_pp(config: dict):
    """
    Create a basic UNet++ model

    Args:
        config:

    Returns:

    """

    # base filter_list = [32, 64, 128, 256, 512]
    filter_list = config["filter_list"]
    upsample_mode = config["upsample_mode"]
    n_class = config["n_class"]
    depth = config["depth"]
    upsample_mode = config["upsample_mode"]
    deep_supervision = config["deep_supervision"]

    model_dict = {}

    # Input layer for the first node
    model_dict["input"] = keras.Input(shape=config["input_dim"], name="x_00_input")
    model_dict["00"] = conv_relu_block(
        node_name="00",
        n_filter=filter_list[0],
        enable_batch_norm=config["enable_batch_norm"],
        mode=upsample_mode,
    )(model_dict["input"])

    # i = indexes the down-sampling layer along the encoder
    # j = indexes the layer in skip connection
    for j in range(depth):
        for i in range(depth):
            if i + j > depth - 1:
                continue

            node_name = node_name_func(i, j)
            print("Creating node ", node_name)

            if j == 0:
                if i != 0:
                    # Downsampling layer
                    layer = layers.MaxPool2D(
                        (2, 2), strides=(2, 2), name=f"x_{node_name}_downsample"
                    )(model_dict[node_name_func(i - 1, j)])
                else:
                    continue
            elif j > 0:
                # Upsampling
                upsample = upsample_layer(
                    node_name=node_name,
                    n_filter=filter_list[i],
                    enable_batch_norm=config["enable_batch_norm"],
                    mode=upsample_mode,
                )(model_dict[node_name_func(i + 1, j - 1)])

                # Get all skip connection
                skip_list = [model_dict[node_name_func(i, k)] for k in range(j)]
                skip_list.append(upsample)

                # Concatenation layer
                layer = layers.Concatenate(name=f"x_{node_name}_concat")(skip_list)

            model_dict[node_name] = conv_relu_block(
                node_name=node_name,
                n_filter=filter_list[i],
                enable_batch_norm=config["enable_batch_norm"],
                mode=upsample_mode,
            )(layer)

    output_lists = []
    activation_dict = {"bin": "sigmoid", "mult": "softmax"}
    for out_name, nc in n_class.items():
        for node_num in range(1, depth):
            model_dict[f"output_{node_num}_{out_name}_c{nc}"] = layers.Conv2D(
                filters=nc,
                kernel_size=1,
                name=f"output_{node_num}_{out_name}_c{nc}",
                padding="same",
                activation=activation_dict.get(out_name, "relu"),
            )(model_dict[f"0{node_num}"])
            output_lists.append(model_dict[f"output_{node_num}_{out_name}_c{nc}"])

    if deep_supervision:
        return keras.Model(inputs=model_dict["input"], outputs=output_lists)

    n_head = len(list(n_class.keys()))
    output_4_index = [i - 1 for i in range(0, (depth - 1) * n_head + 1, depth - 1)]

    return keras.Model(
        inputs=model_dict["input"],
        outputs=output_lists[-1]
        if n_head == 1
        else [output_lists[index] for index in output_4_index[1:]],
    )


def mobile_unetpp(config: dict):
    """
    A UNet++ Model with MobileNetV2 blocks

    Args:
        config:

    Returns:

    """
    filter_list = config["filter_list"]
    downsample_iteration = config["downsample_iteration"]
    upsample_mode = config["upsample_mode"]
    n_class = config["n_class"]
    depth = config["depth"]
    upsample_mode = config["upsample_mode"]
    deep_supervision = config["deep_supervision"]

    model_dict = {}

    model_dict["input"] = keras.Input(shape=config["input_dim"], name="x_00_input")
    model_dict["00"] = sequence_inv_res_bot_block(
        node_name="x_00", filters=filter_list[0], strides=1, t_expansion=6, n=1
    )(model_dict["input"])

    # i = indexes the down-sampling layer along the encoder
    # j = indexes the layer in skip connection
    for j in range(depth):
        for i in range(depth):
            if i + j > depth - 1:
                continue

            node_name = node_name_func(i, j)
            if j == 0:
                if i != 0:
                    # Downsampling layer
                    layer = sequence_inv_res_bot_block(
                        node_name="x_" + node_name + "_down",
                        filters=filter_list[i],
                        strides=2,
                        t_expansion=6,
                        n=downsample_iteration[i],
                    )(model_dict[node_name_func(i - 1, j)])
                else:
                    continue
            elif j > 0:
                # Upsampling
                upsample = upsample_layer(
                    node_name=node_name,
                    n_filter=filter_list[i],
                    enable_batch_norm=config["enable_batch_norm"],
                    mode=upsample_mode,
                )(model_dict[node_name_func(i + 1, j - 1)])

                # Get all skip connection
                skip_list = [model_dict[node_name_func(i, k)] for k in range(j)]
                skip_list.append(upsample)

                # Concatenation layer
                layer = layers.Concatenate(name=f"x_{node_name}_concat")(skip_list)

            model_dict[node_name] = sequence_inv_res_bot_block(
                node_name="x_" + node_name + "_hblock",
                filters=filter_list[i],
                strides=1,
                t_expansion=6,
                n=downsample_iteration[i],
            )(layer)

    output_lists = []
    activation_dict = {"bin": "sigmoid", "mult": "softmax"}
    for out_name, nc in n_class.items():
        for node_num in range(1, depth):
            model_dict[f"output_{node_num}_{out_name}_c{nc}"] = layers.Conv2D(
                filters=nc,
                kernel_size=1,
                name=f"output_{node_num}_{out_name}_c{nc}",
                padding="same",
                activation=activation_dict.get(out_name, "relu"),
            )(model_dict[f"0{node_num}"])
            output_lists.append(model_dict[f"output_{node_num}_{out_name}_c{nc}"])

    if deep_supervision:
        return keras.Model(inputs=model_dict["input"], outputs=output_lists)

    n_head = len(list(n_class.keys()))
    output_4_index = [i - 1 for i in range(0, (depth - 1) * n_head + 1, depth - 1)]

    return keras.Model(
        inputs=model_dict["input"],
        outputs=output_lists[-1]
        if n_head == 1
        else [output_lists[index] for index in output_4_index[1:]],
    )
