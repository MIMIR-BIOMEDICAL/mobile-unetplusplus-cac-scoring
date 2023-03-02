"Module containing function that help the creation of TFRecord"
import pathlib
import sys

# import h5py
# import numpy as np
import tensorflow as tf

sys.path.append(pathlib.Path.cwd().as_posix())

from src.data.preprocess.lib.utils import (  # pylint: disable=wrong-import-position,import-error
    filtered_patient_number_zfill_range, train_test_val_split)


def str_feature(str_input: str):
    """
    A function that convert string input into a
    tensorflow bytes_list feature

    Args:
        str_input:

    Returns:

    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str_input.encode]))


def image_feature(img_input):
    """
    A function that convert a 2D image into a
    tensorflow bytes_list feature

    Args:
        img_input ():

    Returns:

    """
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(img_input).numpy()])
    )


def sparse_coord_feature(sparse_input: list):
    """
    A function that convert a list of list containing the index
    of  sparse array/tensor into a tensorflow int64list feature

    Args:
        sparse_input:

    Returns:

    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=sparse_input.flatten()))


def create_example_fn(combined_data_dict: dict):
    """
    A function that create an example based on the feature schema
    from an input dictionary containing a full schema of one slice
    dataset

    Args:
        combined_data_dict:

    Returns:

    """
    feature = {
        "patient_num": str_feature(combined_data_dict["patient_num"]),
        "idx": str_feature(combined_data_dict["idx"]),
        "img": image_feature(combined_data_dict["img"]),
        "bin_seg": sparse_coord_feature(combined_data_dict["bin_seg"]),
        "mult_seg": sparse_coord_feature(combined_data_dict["mult_seg"]),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def parsed_example_fn(example):
    """
    A function that parsed a serialize example into its original data
    source

    Args:
        example ():

    Returns:

    """
    feature_description = {
        "patient_num": tf.io.FixedLenFeature([], tf.string),
        "idx": tf.io.FixedLenFeature([], tf.string),
        "img": tf.io.FixedLenFeature([], tf.string),
        "bin_seg": tf.io.VarLenFeature(tf.int64),
        "multi_seg": tf.io.VarLenFeature(tf.int64),
    }
    parsed_example = tf.io.parse_example(example, feature_description)
    parsed_example["img"] = tf.io.parse_tensor(parsed_example["img"], tf.uint16)

    dense_bin_seg = tf.sparse.to_dense(parsed_example["bin_seg"])
    parsed_example["bin_seg"] = tf.reshape(dense_bin_seg, [len(dense_bin_seg) // 2, 2])

    dense_multi_seg = tf.sparse.to_dense(parsed_example["multi_seg"])
    parsed_example["mult_seg"] = tf.reshape(
        dense_multi_seg, [len(dense_multi_seg) // 3, 3]
    )
    return parsed_example


def get_patient_split(split_arr: list, random_seed=811):
    """
    A function that create the randomize patient base on
    train,test, val split and also a random_seed

    Args:
        random_seed ():
        split_arr:

    Returns:

    """
    if len(split_arr) != 3:
        raise Exception("Split array should have only 3 member")

    if sum(split_arr) > 1:
        raise Exception("Split array should have the sum equaling to 1")

    calc_patient_arr = filtered_patient_number_zfill_range(0, 450)
    no_calc_patient_arr = filtered_patient_number_zfill_range(451, 789)

    calc_split = train_test_val_split(calc_patient_arr, split_arr, random_seed)
    no_calc_split = train_test_val_split(no_calc_patient_arr, split_arr, random_seed)

    for split_type in ["train", "val", "test"]:
        calc_split[split_type].extend(no_calc_split[split_type])

    return calc_split
