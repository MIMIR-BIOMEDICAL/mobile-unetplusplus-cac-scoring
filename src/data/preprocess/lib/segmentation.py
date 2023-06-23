"""Module for preprocessing segmentation file"""
import pathlib
import plistlib
import sys

import numpy as np
from geo_rasterize import rasterize
from shapely import Polgyon
from tqdm import tqdm

sys.path.append(pathlib.Path.cwd().as_posix())
from src.data.preprocess.lib.utils import (  # pylint: disable=wrong-import-position,import-error
    artery_loc_to_abbr, blacklist_cant_fill, blacklist_invalid_dicom,
    blacklist_mislabelled_roi, blacklist_multiple_image_id_with_roi,
    blacklist_no_image, blacklist_pixel_overlap, convert_abr_to_num,
    fill_segmentation, string_to_float_tuple, string_to_int_tuple)
from src.system.pipeline.output import auto_cac, ground_truth_auto_cac


def convert_plist_to_dict(plist_path: pathlib.Path) -> dict:
    """
    This function convert a plist file (in xml form) to dictionary

    Args:
        path (pathlib.Path): Path to file

    Returns:
        output (dict): A dictionary representation of the input file
    """
    # Check if file have xml extension
    if plist_path.suffix != ".xml":
        raise Exception("File neeeded to be using xml extension")

    # Check if file exists
    if not plist_path.exists():
        raise Exception("File doesn't exist")

    # Get dictionary out of plist file
    with plist_path.open(mode="rb") as file:
        output = plistlib.load(file)

    return output


def clean_raw_segmentation_dict(project_root_path, raw_segmentation_dict: dict) -> dict:
    """
    This function convert a dictionary parsed from raw segmentation json
    to a clean segmentation dictionary

    Args:
        raw_segmentation_dict: dictionary parsed from raw_segmnetation.json

    Returns:
        clean_segmentation_dict: dictionary containing a cleaned segmentation data
    """
    # Output Variable
    clean_output_dict = {}
    patient_no_fill_log = []
    patient_minus_log = []
    patient_agatston_zero = []

    # Transverse dictionary
    for patient_number, patient_image_dict in tqdm(
        raw_segmentation_dict.items(), desc="Cleaning Raw Segmentation JSON"
    ):
        # Blacklist patient due to:
        # - roi pixel overlapping
        # - mislabelled roi
        # - patient with multiple image id and roi
        # - invalid dicom
        # - no image
        if (
            patient_number in blacklist_pixel_overlap()
            or patient_number in blacklist_mislabelled_roi()
            or patient_number in blacklist_multiple_image_id_with_roi()
            or patient_number in blacklist_invalid_dicom()
            or patient_number in blacklist_no_image()
        ):
            continue

        images_list = patient_image_dict["Images"]

        # Check if no image
        if len(images_list) == 0:
            clean_output_dict[patient_number] = {}
            continue

        patient_img_list = []

        for image_dict in images_list:
            cleaned_roi_list = []
            for roi in image_dict["ROIs"]:
                # Check if there is no area or no Px point
                if roi["Area"] == 0 or len(roi["Point_px"]) == 0:
                    continue

                artery_abbreviation = artery_loc_to_abbr(roi["Name"])
                if artery_abbreviation is None:
                    continue

                float_pixel_coord_list = [
                    string_to_float_tuple(string_coord)
                    for string_coord in roi["Point_px"]
                ]

                lesion_polygon = Polgyon(float_pixel_coord_list)
                rasterized_polygon = rasterize(
                    [lesion_polygon], [1], (512, 512), algorithm="replace"
                )

                rasterized_coord = np.argwhere(rasterized_polygon == 1).tolist()

                # Convert string coords to integer coords
                # int_pixel_coord_list = [
                #     string_to_int_tuple(string_coord)
                #     for string_coord in roi["Point_px"]
                # ]

                # Remove duplicate coords
                # pixel_coord_list = list(set(int_pixel_coord_list))

                # Flood fill
                # dense_arr = np.zeros((512, 512))
                # dense_arr[tuple(zip(*pixel_coord_list))] = 1
                # flooded_arr = fill_segmentation(dense_arr)
                # filled_pixel_coord = np.argwhere(flooded_arr == 1).tolist()
                # if len(filled_pixel_coord) <= len(pixel_coord_list):
                #     patient_no_fill_log.append(patient_number)

                cleaned_roi = {"loc": artery_abbreviation, "pos": rasterized_coord}
                cleaned_roi_list.append(cleaned_roi)
            # Skip adding to cleaned data if no roi is detected
            if len(cleaned_roi_list) == 0:
                continue

            # Get patient root path
            if patient_number != "000":
                patient_idx = int(patient_number.lstrip("0"))
            else:
                patient_idx = 0
            patient_root_path = next(project_root_path.rglob(f"patient/{patient_idx}"))

            # Get the amount dicom file in patient folder
            patient_dcm_len = len(list(patient_root_path.rglob("*.dcm")))

            # Image index in metadata is reversed from the actual image index in
            # patient folder, so true index needed to be calculated
            true_image_index = (patient_dcm_len + 1) - image_dict["ImageIndex"]
            if true_image_index < 0:
                patient_minus_log.append(patient_number)
            else:
                # NOTE: Its possible to check Agatston too here, so if no agatston its not worth it
                agatston = ground_truth_auto_cac(
                    [
                        next(
                            patient_root_path.rglob(
                                f"*00{str(true_image_index-1).zfill(2)}.dcm"
                            )
                        )
                    ],
                    [filled_pixel_coord],
                    mem_opt=True,
                )
                if agatston["total_agatston"] == 0:
                    patient_agatston_zero.append(patient_number)

            # NOTE: Check also if there is any off by on eerror

            patient_img_list.append(
                {"idx": str(true_image_index).zfill(3), "roi": cleaned_roi_list}
            )

        clean_output_dict[patient_number] = patient_img_list
    print(set(patient_no_fill_log))
    print(set(patient_minus_log))
    print(set(patient_agatston_zero))

    return clean_output_dict


def split_clean_segmentation_to_binary(clean_segmentation_dict: dict) -> dict:
    """
    A function to get the binary segmentation representation
    of the cleaned segmentation dataset. This binary representation
    is used to train the binary classification head of the model
    so that it can check whether there is a calcium or not

    Args:
        clean_segmentation_dict: dictionary containing the cleaned segmentation dat

    Returns:
        binary_segmentation_dict: dictionary containing the binary
    """
    binary_segmentation_dict = {}
    for patient_number, image_list in tqdm(
        clean_segmentation_dict.items(), desc="Extracting Binary Segmentation Data"
    ):
        out_image_list = []
        for image in image_list:
            image_index = image["idx"]
            roi_list = image["roi"]
            pos_list = []
            for roi in roi_list:
                pos_list.extend(roi["pos"])
            out_image_list.append({"idx": image_index, "pos": pos_list})
        binary_segmentation_dict[patient_number] = out_image_list
    return binary_segmentation_dict


def split_clean_segmentation_to_multiclass(clean_segmentation_dict: dict) -> dict:
    """
    A function to get the multiclas segmentation representation.
    There isn't much difference between the cleaned segmentation data,
    the only difference is in the location (i.e LAD, RCA, etc) that is
    encoded to number, to help the making of a 3D Sparse Matrix. This
    data will be used for the multiclass head of the model

    Args:
        clean_segmentation_dict:

    Returns:

    """
    for _, image_list in tqdm(
        clean_segmentation_dict.items(), desc="Extracting Multiclass Segmentation Data"
    ):
        for image in image_list:
            roi_list = image["roi"]
            for roi in roi_list:
                roi["loc"] = convert_abr_to_num(roi["loc"])
    return clean_segmentation_dict
