"""Module for preprocessing segmentation file"""
import pathlib
import plistlib
import sys

from tqdm import tqdm

sys.path.append(pathlib.Path.cwd().as_posix())

from src.data.preprocess.lib.utils import (  # pylint: disable=wrong-import-position,import-error
    artery_loc_to_abbr, string_to_int_tuple)


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


def clean_raw_segmentation_dict(raw_segmentation_dict: dict) -> dict:
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

    # Transverse dictionary
    for patient_number, patient_image_dict in tqdm(
        raw_segmentation_dict.items(), desc="Cleaning Raw Segmentation JSON"
    ):
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

                # Convert string coords to integer coords
                int_pixel_coord_list = [
                    string_to_int_tuple(string_coord)
                    for string_coord in roi["Point_px"]
                ]

                # Remove duplicate coords
                int_pixel_coord_list = list(set(int_pixel_coord_list))
                cleaned_roi = {
                    "name": artery_abbreviation,
                    "pos": int_pixel_coord_list,
                }
                cleaned_roi_list.append(cleaned_roi)
            # Skip adding to cleaned data if no roi is detected
            if len(cleaned_roi_list) == 0:
                continue

            patient_img_list.append(
                {"idx": image_dict["ImageIndex"], "roi": cleaned_roi_list}
            )
        clean_output_dict[patient_number] = patient_img_list
    return clean_output_dict
