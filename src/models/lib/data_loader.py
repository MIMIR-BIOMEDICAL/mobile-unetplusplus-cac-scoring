"""Module for loading data for modle training"""
import os
import pathlib
import sys
from functools import partial

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

sys.path.append(pathlib.Path.cwd().as_posix())
from src.data.preprocess.lib.tfrecord import parsed_example_fn
from src.models.lib.config import UNetPPConfig


def preprocess_img(image):
    """
    Preprocesses an image by normalizing its pixel values.

    Args:
        image (tf.Tensor): The input image tensor.

    Returns:
        tf.Tensor: The preprocessed image tensor.

    """
    # Normalize image by subtracting the minimum value and dividing by the range
    return (image - tf.reduce_min(image)) / (
        tf.reduce_max(image) - tf.reduce_min(image)
    )


def create_sample(config: UNetPPConfig, features):
    """
    Creates a sample by preprocessing the input image and generating corresponding target segmentation.

    Args:
        config (UNetPPConfig): The configuration object for the UNet++ model.
        features (dict): The input features dictionary containing the image and segmentation information.

    Returns:
        tuple: A tuple containing the preprocessed image tensor, binary segmentation tensor, and multi-class segmentation tensor.

    """
    input_dims = config.input_dim

    # Preprocess input image
    preprocessed_img = tf.reshape(
        preprocess_img(features["img"]),
        [input_dims[0], input_dims[1], input_dims[2]],
    )

    # Prepare binary segmentation tensor
    bin_seg_dim = (input_dims[0], input_dims[1])
    bin_seg = tf.reshape(
        tf.sparse.to_dense(
            tf.sparse.reorder(
                tf.SparseTensor(
                    dense_shape=bin_seg_dim,
                    values=features["segment_val"],
                    indices=features["bin_seg"],
                )
            )
        ),
        [input_dims[0], input_dims[1], 1],
    )

    # Prepare multi-class segmentation tensor
    mult_seg_dim = (input_dims[0], input_dims[1], config.n_class["mult"] - 1)
    mult_seg_indices = tf.subtract(features["mult_seg"], [[0, 0, 1]])
    mult_seg = tf.concat(
        [
            bin_seg,
            tf.reshape(
                (
                    tf.sparse.to_dense(
                        tf.sparse.reorder(
                            tf.SparseTensor(
                                dense_shape=mult_seg_dim,
                                values=features["segment_val"],
                                indices=tf.where(
                                    tf.equal(mult_seg_indices, -1),
                                    mult_seg_indices + 1,
                                    mult_seg_indices,
                                ),
                            )
                        )
                    ),
                ),
                [input_dims[0], input_dims[1], config.n_class["mult"] - 1],
            ),
        ],
        2,  # axis,
    )

    return preprocessed_img, bin_seg, mult_seg


def create_y_data(config: UNetPPConfig, output_layer_name_list, x, y1, y2):
    """
    Creates the target data for training the neural network based on the provided configuration and input data.

    Args:
        config (UNetPPConfig): The configuration object for the UNet++ model.
        output_layer_name_list (list): The list of output layer names.
        x: The input data.
        y1: The binary segmentation data.
        y2: The multiclass segmentation data.

    Returns:
        tuple: A tuple containing the input data and the target data.

    """
    if config.deep_supervision:
        y_list = [y1 for i in range(config.depth - 1)] + [
            y2 for i in range(config.depth - 1)
        ]

        y_dict = {}
        for layer_name, y in zip(output_layer_name_list, y_list):
            y_dict[layer_name] = y
        return x, y_dict

    return x, (y1, y2)


def create_dataset(
    project_root_path: pathlib.Path,
    config: UNetPPConfig,
    output_layer_name_list: list,
    batch_size: int,
    shuffle_size: int,
):
    """
    Creates a dictionary of datasets for training and validation splits by reading and parsing TFRecord files from
    the given project root path. The parsed examples are then converted into samples with corresponding target segmentations
    using the given configuration. The samples are then batched and returned as a dictionary of datasets.

    Args:
    - project_root_path (pathlib.Path): The path to the project root directory.
    - config (UNetPPConfig): An instance of the UNetPPConfig class containing the configuration parameters.
    - output_layer_name_list (list): A list of layer names to be used as output layers in the model.
    - batch_size (int): The size of each batch.
    - shuffle_size (int): The size of shuffle buffer.

    Returns:
    - dataset_dict (dict): A dictionary containing the training and validation datasets, where the keys are "train" and "val",
                           and the values are corresponding TFRecord datasets.
    """

    dataset_dict = {}
    for split in ["train", "val"]:
        tfrecord_path = list(project_root_path.rglob(f"{split}*.tfrecord"))[0]
        tfrecord_path_pattern = tfrecord_path.parent / f"{split}*.tfrecord"
        print(tf.data.Dataset.list_files(tfrecord_path_pattern.as_posix()))
        dataset_dict[split] = (
            tf.data.TFRecordDataset(
                filenames=tf.data.Dataset.list_files(tfrecord_path_pattern.as_posix()),
                compression_type="GZIP",
                num_parallel_reads=tf.data.AUTOTUNE,
            )
            .map(parsed_example_fn, num_parallel_calls=tf.data.AUTOTUNE)
            .map(partial(create_sample, config), num_parallel_calls=tf.data.AUTOTUNE)
            .shuffle(shuffle_size)
            .batch(
                batch_size,
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            .map(
                partial(create_y_data, config, output_layer_name_list),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            .prefetch(tf.data.AUTOTUNE)
        )

    return dataset_dict
