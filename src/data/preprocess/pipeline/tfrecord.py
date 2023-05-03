"""Module for tfrecord pipeline"""
import json
import pathlib
import sys

import click
import h5py
import numpy as np
import tensorflow as tf
from tqdm import tqdm

sys.path.append(pathlib.Path.cwd().as_posix())

from src.data.preprocess.lib.tfrecord import \
    create_example_fn  # pylint: disable=wrong-import-position,import-error
from src.data.preprocess.lib.utils import (  # pylint: disable=wrong-import-position,import-error
    get_patient_split, get_pos_from_bin_list, get_pos_from_mult_list)


def combine_to_tfrecord(
    random_index_dict,
    project_root_path,
    h5_image_index_path,
    binary_json_path,
    multi_json_path,
    split_mode,
    sample_mode=False,
):  # pylint: disable=too-many-arguments,too-many-locals
    """
    A function that essentially combine image and its segmentation
    and also adding them to TFRecord

    Args:
        random_index_dict ():
        project_root_path ():
        h5_image_index_path ():
        binary_json_path ():
        multi_json_path ():
        split_mode ():
        sample_mode ():
    """
    with binary_json_path.open(mode="r") as binary_json_file:
        binary_seg_dict = json.load(binary_json_file)

        with multi_json_path.open(mode="r") as multi_json_file:
            mult_seg_dict = json.load(multi_json_file)

            with h5py.File(h5_image_index_path, "r") as indexer:
                random_patient_index = random_index_dict[split_mode]

                tf_record_path = (
                    project_root_path / "data" / "processed" / f"{split_mode}.tfrecord"
                )

                with tf.io.TFRecordWriter(tf_record_path.as_posix()) as tf_record_file:
                    for patient_index in tqdm(random_patient_index, desc="Patient"):
                        segment_flag = True

                        if sample_mode:
                            try:
                                patient_index_img_list = list(
                                    indexer[patient_index]["img"]
                                )
                            except:  # pylint: disable=bare-except
                                continue
                        else:
                            patient_index_img_list = list(indexer[patient_index]["img"])

                        patient_index_img_segment_list = list(
                            map(lambda x: x["idx"], binary_seg_dict[patient_index])
                        )

                        # check if patient index not in segmentation json
                        if patient_index not in list(binary_seg_dict.keys()):
                            segment_flag = False

                        for img_index in patient_index_img_list:
                            patient_dict = {
                                "patient_num": patient_index,
                                "idx": img_index,
                                "img": indexer[patient_index]["img"][img_index][
                                    "img_hu"
                                ][:],
                            }

                            if (
                                segment_flag
                                and img_index in patient_index_img_segment_list
                            ):
                                patient_dict["bin_seg"] = np.array(
                                    get_pos_from_bin_list(
                                        binary_seg_dict[patient_index], img_index
                                    )
                                )
                                patient_dict["mult_seg"] = np.array(
                                    get_pos_from_mult_list(
                                        mult_seg_dict[patient_index], img_index
                                    )
                                )
                                patient_dict["segment_val"] = np.ones(
                                    patient_dict["mult_seg"].shape[0]
                                )
                            else:
                                patient_dict["bin_seg"] = np.array([[0, 0]])
                                patient_dict["mult_seg"] = np.array([[0, 0, 0]])
                                patient_dict["segment_val"] = np.zeros(
                                    patient_dict["mult_seg"].shape[0]
                                )

                            example = create_example_fn(patient_dict)
                            tf_record_file.write(example.SerializeToString())


@click.command()
@click.option("-s", "--sample", type=click.BOOL, help="Sample Mode", default=False)
@click.option(
    "-t", "--split_type", type=click.STRING, help="Type of split", default="all"
)
@click.option(
    "-d",
    "--distribution",
    type=click.STRING,
    help="Distribution of train,test and val",
    default="721",
)
def preprocess_tfrecord_pipeline(sample, split_type, distribution):
    """
    A wrapper around the tfrecord creation function
    to be used with click

    Args:
        sample ():
        split_type ():
        distribution ():
    """
    if len(distribution) != 3:
        raise ValueError("Distribution can only contain 3 string")
    if sum((int(n) for n in distribution)) != 10:
        raise ValueError("Total Distribution should be 10")
    project_root_path = pathlib.Path.cwd()
    random_index_dict = get_patient_split([float(n) / 10 for n in distribution])
    h5_image_index_path = list(project_root_path.rglob("index.h5"))[0]
    binary_json_path = list(project_root_path.rglob("binary*.json"))[0]
    multi_json_path = list(project_root_path.rglob("multi*.json"))[0]

    if split_type == "all":
        for split in ["train", "val", "test"]:
            combine_to_tfrecord(
                random_index_dict,
                project_root_path,
                h5_image_index_path,
                binary_json_path,
                multi_json_path,
                split,
                sample,
            )
    else:
        combine_to_tfrecord(
            random_index_dict,
            project_root_path,
            h5_image_index_path,
            binary_json_path,
            multi_json_path,
            split_type,
            sample,
        )


if __name__ == "__main__":
    preprocess_tfrecord_pipeline()  # pylint: disable=no-value-for-parameter
