"""Module containing the base ml model"""
import pathlib
import sys

from tensorflow import \
    keras  # pylint: disable=wrong-import-position,import-error
from tensorflow.keras import \
    layers  # pylint: disable=wrong-import-position,import-error
from tensorflow.keras.applications import MobileNetV2

sys.path.append(pathlib.Path.cwd().as_posix())

from src.models.lib.block import (  # pylint: disable=wrong-import-position,import-error
    conv_bn_relu_block, sequence_inv_res_bot_block, upsample_block)
from src.models.lib.config import UNetPPConfig
from src.models.lib.utils import \
    node_name_func  # pylint: disable=wrong-import-position,import-error


def base_unet_pp(config: UNetPPConfig):
    """
    Build a UNet++ model based on the given configuration.

    Loop base on network connectivity function (1) in UNet++ Paper
    i = indexes the down-sampling layer along the encoder
    j = indexes the layer in skip connection
    H = Convolution operation followed by an activation function
    D = Downsample
    U = Upsample
    [] = Concatenation layer

    j=0
    x(i,j)=H(D(x(i-1,j)))

    j>0
    x(i,j)=H([[x(i,k)](k=0 -> j-1),U(x(i+1,j-1))])

    Args:
        config (UNetPPConfig): Configuration object for the UNet++ model.

    Returns:
        keras.Model: A compiled UNet++ model.
        list: A list containing the output layer name, used for building later

    """
    model_mode_mapping = {
        "basic": {
            "h_block": (
                conv_bn_relu_block,
                {
                    "mode": config.upsample_mode,
                },
            ),
        },
        "mobile": {
            "h_block": (
                sequence_inv_res_bot_block,
                {
                    "strides": 1,
                    "t_expansion": 6,
                    "n_iter": config.downsample_iteration[0]
                    if config.downsample_iteration is not None
                    else 1,
                },
            ),
        },
    }
    model_dict = {}

    # Input layer for the first node
    model_dict["input"] = keras.Input(shape=config.input_dim, name="x_00_input")

    # For the first node it is a H block without downsampling

    # H block initialization
    h_block, h_params = model_mode_mapping[config.model_mode]["h_block"]
    h_params["batch_norm"] = config.batch_norm
    h_params["n_filter"] = config.filter_list[0]

    model_dict["00"] = h_block(node_name="00", **h_params)(model_dict["input"])
    print(f"--- Creating model input node X00 f{config.filter_list[0]}")

    for j in range(config.depth):
        for i in range(max(0, config.depth - j)):
            node_name = node_name_func(i, j)

            if j == 0 and i != 0:
                print(f"--- Creating model downsample node X{node_name} ")
                # Downsampling layer
                down_layer_name = f"{node_name}_down"
                down_layer_input = model_dict[node_name_func(i - 1, j)]

                if config.model_mode == "basic":
                    layer = layers.MaxPool2D(
                        (2, 2), strides=(2, 2), name=f"x_{down_layer_name}"
                    )(down_layer_input)
                elif config.model_mode == "mobile":
                    layer = sequence_inv_res_bot_block(
                        node_name=down_layer_name,
                        batch_norm=config.batch_norm,
                        n_filter=config.filter_list[i],
                        strides=2,
                        t_expansion=6,
                        n_iter=config.downsample_iteration[i],
                    )(down_layer_input)

            elif j > 0:
                print(
                    f"--- Creating model upsample node X{node_name} f {config.filter_list[i]}"
                )
                # Upsampling
                upsample = upsample_block(
                    node_name=node_name,
                    n_filter=config.filter_list[i],
                    batch_norm=config.batch_norm,
                    mode=config.upsample_mode,
                )(model_dict[node_name_func(i + 1, j - 1)])

                # Get all skip connection
                skip_list = [model_dict[node_name_func(i, k)] for k in range(j)]
                skip_list.append(upsample)

                # Concatenation layer
                layer = layers.Concatenate(name=f"x_{node_name}_concat")(skip_list)

            else:  # j==0 and i==0
                continue

            # Add H Block
            h_params["n_filter"] = config.filter_list[i]
            if config.model_mode == "mobile":
                h_params["n_iter"] = config.downsample_iteration[i]
            model_dict[node_name] = h_block(node_name=node_name, **h_params)(layer)

    # Preparation whether we use deep supervision or not
    # This loop basicaclly make sure that a multihead deep supervision is possible

    output_lists = []
    output_layer_name = []
    activation_dict = {"bin": "sigmoid", "mult": "softmax"}

    # Create a bunch of Conv 1x1 to the node with j = 0
    for out_name, nc in config.n_class.items():
        for node_num in range(1, config.depth):
            layer_name = f"ds_{node_num}"
            model_dict[layer_name] = layers.Conv2D(
                filters=nc,
                kernel_size=1,
                name=layer_name,
                padding="same",
                activation=activation_dict.get(out_name, "sigmoid"),
                dtype="float32",
            )(model_dict[f"0{node_num}"])
            output_lists.append(model_dict[layer_name])
            output_layer_name.append(layer_name)

    if config.deep_supervision:
        return (
            keras.Model(inputs=model_dict["input"], outputs=output_lists),
            output_layer_name,
        )

    # Get the index of the last node, with the added head
    n_head = len(list(config.n_class.keys()))
    non_deep_supervision_output_index = [
        i - 1 for i in range(0, (config.depth - 1) * n_head + 1, config.depth - 1)
    ]

    return (
        keras.Model(
            inputs=model_dict["input"],
            outputs=output_lists[-1]
            if n_head == 1
            else [
                output_lists[index] for index in non_deep_supervision_output_index[1:]
            ],
        ),
        output_layer_name[-1]
        if n_head == 1
        else [
            output_layer_name[index] for index in non_deep_supervision_output_index[1:]
        ],
    )


def unetpp_mobile_backend(config: UNetPPConfig):
    model_mode_mapping = {
        "basic": {
            "h_block": (
                conv_bn_relu_block,
                {
                    "mode": config.upsample_mode,
                },
            ),
        },
        "mobile": {
            "h_block": (
                sequence_inv_res_bot_block,
                {
                    "strides": 1,
                    "t_expansion": 6,
                    "n_iter": config.downsample_iteration[0]
                    if config.downsample_iteration is not None
                    else 1,
                },
            ),
        },
    }

    # H block initialization
    h_block, h_params = model_mode_mapping[config.model_mode]["h_block"]
    h_params["batch_norm"] = config.batch_norm

    model_dict = {}

    # For the first node it is a H block without downsampling

    # Input layer for the first node
    model_dict["input"] = keras.Input(shape=config.input_dim, name="x_00_input")
    model_dict["00"] = layers.Concatenate(name="x_00_concat")(
        [model_dict["input"], model_dict["input"], model_dict["input"]]
    )
    mobile_backbone = MobileNetV2(
        input_tensor=model_dict["00"], weights="imagenet", include_top=False
    )

    mobile_skip_node = {
        "10": "block_1_expand_relu",
        "20": "block_3_expand_relu",
        "30": "block_6_expand_relu",
        "40": "block_13_expand_relu",
    }

    for i in range(1, config.depth):
        num = f"{i}0"
        print(f"--- Creating model downsample node X{num} ")
        h_params["n_filter"] = config.filter_list[i]
        if config.model_mode == "mobile":
            h_params["n_iter"] = config.downsample_iteration[i]
        backbone_output = mobile_backbone.get_layer(mobile_skip_node[num]).output
        model_dict[num] = h_block(node_name=num, **h_params)(backbone_output)

    for j in range(config.depth):
        for i in range(max(0, config.depth - j)):
            node_name = node_name_func(i, j)

            if j > 0:
                print(
                    f"--- Creating model upsample node X{node_name} f {config.filter_list[i]}"
                )
                # Upsampling
                upsample = upsample_block(
                    node_name=node_name,
                    n_filter=config.filter_list[i],
                    batch_norm=config.batch_norm,
                    mode=config.upsample_mode,
                )(model_dict[node_name_func(i + 1, j - 1)])

                # Get all skip connection
                skip_list = [model_dict[node_name_func(i, k)] for k in range(j)]
                skip_list.append(upsample)

                # Concatenation layer
                layer = layers.Concatenate(name=f"x_{node_name}_concat")(skip_list)
            else:
                continue

            # Add H Block
            h_params["n_filter"] = config.filter_list[i]
            if config.model_mode == "mobile":
                h_params["n_iter"] = config.downsample_iteration[i]
            model_dict[node_name] = h_block(node_name=node_name, **h_params)(layer)

    # Preparation whether we use deep supervision or not
    # This loop basicaclly make sure that a multihead deep supervision is possible

    output_lists = []
    output_layer_name = []
    activation_dict = {"bin": "sigmoid", "mult": "softmax"}

    # Create a bunch of Conv 1x1 to the node with j = 0
    for out_name, nc in config.n_class.items():
        for node_num in range(1, config.depth):
            layer_name = f"ds_{node_num}"
            model_dict[layer_name] = layers.Conv2D(
                filters=nc,
                kernel_size=1,
                name=layer_name,
                padding="same",
                activation=activation_dict.get(out_name, "sigmoid"),
                dtype="float32",
            )(model_dict[f"0{node_num}"])
            output_lists.append(model_dict[layer_name])
            output_layer_name.append(layer_name)

    if config.deep_supervision:
        return (
            keras.Model(inputs=model_dict["input"], outputs=output_lists),
            output_layer_name,
        )

    # Get the index of the last node, with the added head
    n_head = len(list(config.n_class.keys()))
    non_deep_supervision_output_index = [
        i - 1 for i in range(0, (config.depth - 1) * n_head + 1, config.depth - 1)
    ]

    return (
        keras.Model(
            inputs=model_dict["input"],
            outputs=output_lists[-1]
            if n_head == 1
            else [
                output_lists[index] for index in non_deep_supervision_output_index[1:]
            ],
        ),
        [output_layer_name[-1]]
        if n_head == 1
        else [
            output_layer_name[index] for index in non_deep_supervision_output_index[1:]
        ],
    )
