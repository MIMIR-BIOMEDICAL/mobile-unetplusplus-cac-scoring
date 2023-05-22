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

from src.data.preprocess.lib.tfrecord import (  # pylint: disable=wrong-import-position,import-error
    create_example_fn,
)
from src.data.preprocess.lib.utils import (  # pylint: disable=wrong-import-position,import-error
    fill_segmentation,
    get_patient_split,
    get_pos_from_bin_list,
    get_pos_from_mult_list,
    split_list,
)


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

                # NOTE: 1. create a sharded list of random_patient index
                # NOTE: 2. Use enumerate to get each index of sharding + zfill i guess
                # NOTE: 3. Add TFRecordOptions with compression type gzip and compression level 9
                # NOTE: 4. Use it in TFRecordWriter

                random_patient_index_shard = split_list(random_patient_index, 10)

                for shard_count, patient_shard in enumerate(
                    tqdm(
                        random_patient_index_shard,
                        desc=f"{split_mode} TFRecord Sharding",
                        unit="shard",
                    )
                ):
                    tf_record_path = (
                        project_root_path
                        / "data"
                        / "processed"
                        / f"{split_mode}-{str(shard_count).zfill(4)}.tfrecord"
                    )
                    with tf.io.TFRecordWriter(
                        tf_record_path.as_posix(),
                        options=tf.io.TFRecordOptions(
                            compression_type="GZIP", compression_level=9
                        ),
                    ) as tf_record_file:
                        for patient_index in tqdm(
                            patient_shard,
                            desc="Patient",
                            unit="patient",
                            leave=False,
                        ):
                            if sample_mode:
                                try:
                                    patient_index_img_list = list(
                                        indexer[patient_index]["img"]
                                    )
                                except:  # pylint: disable=bare-except
                                    continue
                            else:
                                patient_index_img_list = list(
                                    indexer[patient_index]["img"]
                                )

                            for img_index in tqdm(
                                patient_index_img_list,
                                desc="Image",
                                unit="image",
                                leave=False,
                            ):
                                patient_dict = {
                                    "patient_num": patient_index,
                                    "idx": img_index,
                                    "img": indexer[patient_index]["img"][img_index][
                                        "img_hu"
                                    ][:],
                                }

                                # Add segmentation if patient index in bin_seg_dict keys (check if patient calc)
                                # and the current img index is calc (check if image calc)
                                if patient_index in list(
                                    binary_seg_dict.keys()
                                ) and img_index in list(
                                    map(
                                        lambda x: x["idx"],
                                        binary_seg_dict.get(
                                            patient_index, [{"idx": -1}]
                                        ),
                                    )
                                ):
                                    raw_bin_seg = np.array(
                                        get_pos_from_bin_list(
                                            binary_seg_dict[patient_index], img_index
                                        )
                                    )
                                    raw_mult_seg = np.array(
                                        get_pos_from_mult_list(
                                            mult_seg_dict[patient_index], img_index
                                        )
                                    )

                                    dense_bin = np.zeros((512, 512))
                                    dense_bin[tuple(zip(*raw_bin_seg))] = 1
                                    flooded_bin = fill_segmentation(dense_bin)
                                    patient_dict["bin_seg"] = np.argwhere(
                                        flooded_bin == 1
                                    )

                                    dense_mult = np.zeros((512, 512, 5))
                                    flooded_mult = np.zeros((512, 512, 5))
                                    dense_mult[tuple(zip(*raw_mult_seg))] = 1
                                    for i in range(1, 5):
                                        flooded_mult[:, :, i] = fill_segmentation(
                                            dense_mult[:, :, i]
                                        )
                                    patient_dict["mult_seg"] = np.argwhere(
                                        flooded_mult == 1
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
