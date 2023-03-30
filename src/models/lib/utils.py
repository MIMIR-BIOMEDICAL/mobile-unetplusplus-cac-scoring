"""Module containing utility function for UNet++ model"""
import pathlib
import sys
from typing import Dict, List, Union

import tensorflow as tf

sys.path.append(pathlib.Path.cwd().as_posix())

from src.models.lib.config import UNetPPConfig


def node_name_func(i: int, j: int) -> str:
    """
    Generates a node name based on two integer inputs.

    Args:
        i (int): The first integer input.
        j (int): The second integer input.

    Returns:
        str: A string containing the concatenation of i and j.
    """
    return str(i) + str(j)


def loss_dict_gen(
    config: UNetPPConfig,
    output_name_list: List[str],
    loss_list: List[Union[str, tf.keras.losses.Loss]],
) -> Dict[str, Union[str, tf.keras.losses.Loss]]:
    """
    Generates a dictionary of loss functions for each output head.

    Args:
        config (UNetPPConfig): Configuration for the model.
        output_name_list (List[str]): A list of names for the output heads.
        loss_list (List[Union[str, tf.keras.losses.Loss]]): A list of loss functions or their names for each output head.

    Returns:
        A dictionary with the output head names as keys and their corresponding loss functions as values.
    Raises:
        ValueError: If the length of the loss_list is not the same as the number of output head classes.
    """

    if len(config.n_class.keys()) != len(loss_list):
        raise ValueError(
            "Loss list must have the same length as the number of output head classes"
        )

    out_dict = {}

    if config.deep_supervision:
        factor = len(output_name_list) // len(loss_list)
        for i, item in enumerate(output_name_list):
            out_dict[item] = loss_list[i // factor]
    else:
        for name, loss in zip(output_name_list, loss_list):
            out_dict[name] = loss

    return out_dict
