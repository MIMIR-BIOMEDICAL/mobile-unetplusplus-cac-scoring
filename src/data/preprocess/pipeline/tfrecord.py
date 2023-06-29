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
    artery_loc_to_abbr,
    blacklist_agatston_zero,
    blacklist_invalid_dicom,
    blacklist_mislabelled_roi,
    blacklist_multiple_image_id,
    blacklist_multiple_image_id_with_roi,
    blacklist_neg_reverse_index,
    blacklist_no_image,
    blacklist_pixel_overlap,
    convert_abr_to_num,
    fill_segmentation,
    get_patient_split,
    get_pos_from_bin_list,
    get_pos_from_mult_list,
    split_list,
    string_to_float_tuple,
    string_to_int_tuple,
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
    np.random.seed(811)
    log = {}
    with open("result.json") as json_file:
        agatston_map = json.load(json_file)

    with binary_json_path.open(mode="r") as binary_json_file:
        binary_seg_dict = json.load(binary_json_file)

        with multi_json_path.open(mode="r") as multi_json_file:
            mult_seg_dict = json.load(multi_json_file)

            with h5py.File(h5_image_index_path, "r") as indexer:
                random_patient_index = random_index_dict[split_mode]
                print(split_mode, len(random_patient_index))

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
                            compression_type="GZIP", compression_level=6
                        ),
                    ) as tf_record_file:
                        for patient_index in tqdm(
                            patient_shard,
                            desc="Patient",
                            unit="patient",
                            leave=False,
                        ):
                            if (
                                patient_index in blacklist_pixel_overlap()
                                or patient_index in blacklist_mislabelled_roi()
                                or patient_index in blacklist_multiple_image_id()
                                or patient_index in blacklist_invalid_dicom()
                                or patient_index in blacklist_no_image()
                                or patient_index in blacklist_neg_reverse_index()
                                or patient_index in blacklist_agatston_zero()
                            ):
                                continue
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

                            cl = agatston_map.get(
                                patient_index, {"total_agatston": 0, "class": "Absent"}
                            )["class"]
                            log[cl] = log.get(cl, 0) + 1

                            for img_index in tqdm(
                                patient_index_img_list,
                                desc="Image",
                                unit="image",
                                leave=False,
                            ):
                                patient_dict = {}
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

                                    img_is_cac = True
                                else:
                                    img_is_cac = False
                                    patient_dict["bin_seg"] = np.array([[0, 0]])
                                    patient_dict["mult_seg"] = np.array([[0, 0, 0]])
                                    patient_dict["segment_val"] = np.zeros(
                                        patient_dict["mult_seg"].shape[0]
                                    )

                                patient_dict["patient_num"] = patient_index
                                patient_dict["idx"] = img_index

                                if img_is_cac:
                                    log_key = f"{split_mode}-img-cac"
                                    log[log_key] = log.get(log_key, 0) + 1
                                    log[log_key + " cac_pixel"] = (
                                        log.get(log_key + " cac_pixel", 0)
                                        + patient_dict["mult_seg"].shape[0]
                                    )
                                    log[log_key + " non_cac_pixel"] = (
                                        log.get(log_key + " non_cac_pixel", 0)
                                        + 512 * 512
                                        - patient_dict["mult_seg"].shape[0]
                                    )
                                    # patient_dict["img"] = indexer[patient_index]["img"][
                                    #     img_index
                                    # ]["img_hu"][:]
                                    #
                                    # example = create_example_fn(patient_dict)
                                    # tf_record_file.write(example.SerializeToString())
                                else:
                                    log_key = f"{split_mode}-img-non-cac"
                                    if split_mode == "train":
                                        log[log_key] = log.get(log_key, 0) + 1
                                        log[log_key + " non_cac_pixel"] = (
                                            log.get(
                                                log_key + " non_cac_pixel", 0
                                            )
                                            + 512 * 512
                                        )
                                        # diff = 1984 - log.get(log_key, 0)
                                        #
                                        # if diff <= 0:
                                        #     continue
                                        # else:
                                        #     skip = np.random.choice(
                                        #         2, size=1, p=[0.91, 0.09]
                                        #     )[0]
                                        #
                                        #     if skip:
                                        #         continue
                                        #     else:
                                        #         log[log_key] = log.get(log_key, 0) + 1
                                        #         log[log_key + " non_cac_pixel"] = (
                                        #             log.get(
                                        #                 log_key + " non_cac_pixel", 0
                                        #             )
                                        #             + 512 * 512
                                        #         )
                                                # patient_dict["img"] = indexer[
                                                #     patient_index
                                                # ]["img"][img_index]["img_hu"][:]
                                                # #
                                                # example = create_example_fn(
                                                #     patient_dict
                                                # )
                                                # tf_record_file.write(
                                                #     example.SerializeToString()
                                                # )
                                    else:
                                        log[log_key] = log.get(log_key, 0) + 1
                                        log[log_key + " non_cac_pixel"] = (
                                            log.get(log_key + " non_cac_pixel", 0)
                                            + 512 * 512
                                        )
                                        # patient_dict["img"] = indexer[patient_index][
                                        #     "img"
                                        # ][img_index]["img_hu"][:]
                                        # example = create_example_fn(patient_dict)
                                        # tf_record_file.write(
                                        #     example.SerializeToString()
                                        # )

                                # Over sample algorithmm
                                # CAC = 2391
                                # if img_is_cac:
                                #     log_key = f"{split_mode}-img-cac"
                                #     if split_mode == "train":
                                #         diff = 12112 - log.get(log_key, 0)
                                #
                                #         if diff <= cac and diff > 0:
                                #             diff = 1
                                #
                                #         n_loop = np.random.choice(
                                #             np.arange(3, 10), size=1
                                #         )[0]
                                #
                                #         n_loop_min = np.min([diff, n_loop])
                                #         log[n_loop_min] = log.get(n_loop_min, 0) + 1
                                #
                                #         for _ in range(n_loop_min):
                                #             log[log_key] = log.get(log_key, 0) + 1
                                #             tf_record_file.write(
                                #                 example.SerializeToString()
                                #             )
                                #         cac -= 1
                                #     else:
                                #         log[log_key] = log.get(log_key, 0) + 1
                                #         tf_record_file.write(
                                #             example.SerializeToString()
                                #         )
                                # else:
                                #     log_key = f"{split_mode}-img-non-cac"
                                #     log[log_key] = log.get(log_key, 0) + 1
                                #     tf_record_file.write(example.SerializeToString())

    print(log)


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
