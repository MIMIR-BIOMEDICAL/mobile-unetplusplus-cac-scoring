"""Module containing utility function for UNet++ model"""
import pathlib
import sys
from typing import Callable, Dict, List, Union

import numpy as np
import tensorflow as tf
from skimage.segmentation import flood_fill

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
    loss_list: List[Union[str, tf.keras.losses.Loss, Callable]],
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


def parse_list_string(list_string):
    """
    Parses a string representation of a list of integers, separated by commas,
    into a Python list of integers.

    Args:
        list_string (str): A string representation of a list of integers, e.g.
            "512,512,1".

    Returns:
        list: A list of integers parsed from the input string, e.g. [512, 512, 1].
    """
    out_list = list_string.split(",")
    out_list = [int(num) for num in out_list]
    return out_list


def flood_fill_scanline(image, start_coord, new_value):
    """
    Performs flood fill using scanline algorithm on a 2D image.

    Args:
        image (numpy.ndarray): Input image as a 2D NumPy array.
        start_coord (tuple): Starting coordinate (x, y) for flood fill.
        new_value: Value to fill in the flooded area.

    Returns:
        numpy.ndarray: Flood-filled image.
    """
    rows, cols = image.shape
    stack = [
        (start_coord[0], start_coord[1])
    ]  # Initialize the stack with the starting coordinate
    start_value = image[
        start_coord[0], start_coord[1]
    ]  # Get the value at the starting coordinate

    if start_value == new_value:
        return image

    while stack:
        x, y = stack.pop()  # Pop the next coordinate from the stack
        if image[x, y] != start_value:
            continue  # Skip if the current pixel does not have the start value

        left, right = y, y
        while left >= 0 and image[x, left] == start_value:
            left -= 1  # Find the left boundary of the flood area
        while right < cols and image[x, right] == start_value:
            right += 1  # Find the right boundary of the flood area

        image[x, left + 1 : right] = new_value  # Fill the flood area with the new value

        if x > 0:
            for i in range(left + 1, right):
                if image[x - 1, i] == start_value:
                    stack.append(
                        (x - 1, i)
                    )  # Add neighboring pixels from the above row to the stack

        if x < rows - 1:
            for i in range(left + 1, right):
                if image[x + 1, i] == start_value:
                    stack.append(
                        (x + 1, i)
                    )  # Add neighboring pixels from the below row to the stack

    return image


def fill_segmentation(segmentation, mode="skimage"):
    """
    Fill the segmentation using flood fill algorithm.

    Args:
        segmentation (tf.Tensor): Input segmentation as a 2D TensorFlow tensor.
        mode (str): Mode for flood fill algorithm. Default is "skimage".

    Returns:
        tf.Tensor: Filled segmentation.

    Raises:
        ValueError: If an unsupported mode is provided.
    """

    squeeze_segmentation = tf.squeeze(segmentation)
    # Convert to NumPy array
    np_segmentation = squeeze_segmentation.numpy()
    if mode == "skimage":
        # Use flood_fill function from skimage
        flood_filled_segmentation = flood_fill(
            np_segmentation, (0, 0), 2, connectivity=1
        )
    elif mode == "scanline":
        # Use custom scanline flood fill algorithm
        flood_filled_segmentation = flood_fill_scanline(np_segmentation, (0, 0), 2)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    flipped_segmentation = np.where(
        flood_filled_segmentation == 2,
        0,
        np.where(flood_filled_segmentation == 0, 1, flood_filled_segmentation),
    )

    return tf.convert_to_tensor(flipped_segmentation, dtype=segmentation.dtype)
