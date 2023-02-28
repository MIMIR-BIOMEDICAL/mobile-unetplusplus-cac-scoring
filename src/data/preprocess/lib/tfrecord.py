"Module containing function that help the creation of TFRecord"
# import pathlib
# import sys

# import h5py
# import numpy as np
import tensorflow as tf

# sys.path.append(pathlib.Path.cwd().as_posix())
#
# from src.data.preprocess.lib.utils import (  # pylint: disable=wrong-import-position,import-error
#     blacklist_invalid_dicom, blacklist_mislabelled_roi,
#     blacklist_multiple_image_id, blacklist_no_image, blacklist_pixel_overlap,
#     patient_number_zfill_range)


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
