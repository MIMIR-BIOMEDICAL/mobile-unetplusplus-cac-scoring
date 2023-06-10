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

    # Clip HU [-800,1200]
    clipped_image = tf.clip_by_value(image, -800, 1200)

    # Normalization
    normalized_image = (clipped_image - -800) / (1200 - -800)

    # Zero Centering
    zero_centered_image = normalized_image - tf.reduce_mean(normalized_image)

    return zero_centered_image


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
    preprocessed_img = preprocess_img(features["img"])
    preprocessed_img = tf.reshape(preprocessed_img, input_dims)

    # Prepare binary segmentation tensor
    bin_seg_dim = (input_dims[0], input_dims[1])
    bin_seg = tf.SparseTensor(
        dense_shape=bin_seg_dim,
        values=features["segment_val"],
        indices=features["bin_seg"],
    )
    bin_seg = tf.sparse.reorder(bin_seg)
    bin_seg = tf.sparse.to_dense(bin_seg)
    bin_seg = tf.reshape(bin_seg, [input_dims[0], input_dims[1], 1])

    # Prepare multi-class segmentation tensor
    mult_seg_dim = (input_dims[0], input_dims[1], config.n_class["mult"] - 1)
    mult_seg_indices = tf.subtract(features["mult_seg"], [[0, 0, 1]])

    # Convert bin_seg to mult_seg
    bin_to_mult = tf.cast(tf.where(bin_seg == 0, 1, 0), tf.float32)

    # Convert mult_seg_indices to dense mult_seg
    converted_indices = tf.where(
        tf.equal(mult_seg_indices, -1),
        mult_seg_indices + 1,
        mult_seg_indices,
    )
    dense_mult_seg = tf.SparseTensor(
        dense_shape=mult_seg_dim,
        values=features["segment_val"],
        indices=converted_indices,
    )
    dense_mult_seg = tf.sparse.reorder(dense_mult_seg)
    dense_mult_seg = tf.sparse.to_dense(dense_mult_seg)

    # Concatenate bin_to_mult and dense_mult_seg along the last dimension
    mult_seg = tf.concat(
        [
            bin_to_mult,
            tf.reshape(dense_mult_seg, mult_seg_dim),
        ],
        axis=2,
    )

    # Fix row column and x y mix up
    mult_seg = tf.transpose(mult_seg, perm=(1, 0, 2))

    return preprocessed_img, tf.cast(mult_seg, tf.float32)


def create_y_data(config: UNetPPConfig, x, y):
    """
    Creates the target data for training the neural network based on the provided configuration and input data.

    Args:
        config (UNetPPConfig): The configuration object for the UNet++ model.
        x: The input data.
        y: The multiclass segmentation data.

    Returns:
        tuple: A tuple containing the input data and the target data.

    """
    if config.deep_supervision:
        y_copies = tuple(tf.identity(y) for _ in range(config.depth - 1))
        return x, y_copies

    return x, y


def create_dataset(
    project_root_path: pathlib.Path,
    config: UNetPPConfig,
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
    for split in ["train", "val", "test"]:
        tfrecord_path = list(project_root_path.rglob(f"{split}*.tfrecord"))[0]
        tfrecord_path_pattern = tfrecord_path.parent / f"{split}*.tfrecord"
        dataset = (
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
                drop_remainder=True,
            )
            .map(
                partial(create_y_data, config),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            .prefetch(tf.data.AUTOTUNE)
        )
        dataset_dict[split] = dataset

    return dataset_dict
