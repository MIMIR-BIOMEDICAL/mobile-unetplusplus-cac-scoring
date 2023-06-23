"""Library module for image preprocessing"""
import pathlib
import sys

from tqdm import tqdm

sys.path.append(pathlib.Path.cwd().as_posix())

from src.data.preprocess.lib.utils import (  # pylint: disable=wrong-import-position,import-error
    artery_loc_to_abbr, blacklist_agatston_zero, blacklist_invalid_dicom,
    blacklist_mislabelled_roi, blacklist_multiple_image_id,
    blacklist_multiple_image_id_with_roi, blacklist_neg_reverse_index,
    blacklist_no_image, blacklist_pixel_overlap, convert_abr_to_num,
    fill_segmentation, string_to_float_tuple, string_to_int_tuple)


def extract_patient_dicom_path(gated_path: pathlib.Path):
    """
    A function that extract the path of all dicom file
    and group them base on its patient number

    Args:
        gated_path:

    Returns:

    """
    dicom_path = list(gated_path.rglob("*.dcm"))
    len_dicom_path = len(list(dicom_path))
    dicom_path = list(gated_path.rglob("*.dcm"))
    patient_image_dict = {}

    for path in tqdm(dicom_path, desc="Extracting DICOM Path", total=len_dicom_path):
        patient_number = str(path.parent.parent.stem).zfill(3)

        # Blacklist patient
        # - image with overlapping roi
        # - image with mislabelled roi
        # - image with multiple dataset
        # - invalid dicom
        if (
            patient_number in blacklist_pixel_overlap()
            or patient_number in blacklist_mislabelled_roi()
            or patient_number in blacklist_multiple_image_id()
            or patient_number in blacklist_invalid_dicom()
            or patient_number in blacklist_no_image()
            or patient_number in blacklist_neg_reverse_index()
            or patient_number in blacklist_agatston_zero()
        ):
            continue

        image_index = path.stem[-3:]

        if patient_image_dict.get(patient_number) is None:
            patient_image_dict[patient_number] = []

        patient_image_dict[patient_number].append({"idx": image_index, "path": path})

    # Sort patient number
    patient_image_dict_key = list(patient_image_dict.keys())
    sorted_patient_image_dict_key = sorted(patient_image_dict_key)

    sorted_patient_image_dict = {
        index: patient_image_dict[index] for index in sorted_patient_image_dict_key
    }

    # Sort per patient image index
    for patient_number in sorted_patient_image_dict.keys():
        sorted_patient_image_dict[patient_number] = sorted(
            sorted_patient_image_dict[patient_number], key=lambda x: x["idx"]
        )

    return sorted_patient_image_dict
