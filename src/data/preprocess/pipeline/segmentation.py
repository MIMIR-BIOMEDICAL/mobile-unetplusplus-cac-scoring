"""Module containing the pipeline for preprocessing segmentation file"""
import json
import pathlib
import sys

from tqdm import tqdm

sys.path.append(pathlib.Path.cwd().as_posix())

from src.data.preprocess.lib.segmentation import (  # pylint: disable=import-error,wrong-import-position
    clean_raw_segmentation_dict,
    convert_plist_to_dict,
    split_clean_segmentation_to_binary,
    split_clean_segmentation_to_multiclass,
)


def create_raw_segmentation_json(project_root_path: pathlib.Path):
    """
    A part of pipeline function that convert all of segmentation
    plist file to json. No prior preprocessing is done

    Args:
        project_root_path: Path to the main working directory
    """

    # Find xml all plist file inside data
    raw_xml_file_list = project_root_path.rglob("*.xml")
    xml_file_list = filter(
        lambda file: "-checkpoint" not in str(file), raw_xml_file_list
    )

    # Variable used for output
    out_dict = {}

    # Make a dictionary with zfilled patient number as key
    # and the corresponding raw converted plist dictionary as value
    for xml_file in tqdm(xml_file_list, desc="Converting Segmentation Plist to JSON"):
        file_name = xml_file.stem
        zfill_file_name = file_name.zfill(3)

        convert_dict = convert_plist_to_dict(xml_file)

        out_dict[zfill_file_name] = convert_dict

    # Sort dictionary base on patient number (low to high)
    out_dict_key = list(out_dict.keys())
    sorted_out_dict_key = sorted(out_dict_key)

    sorted_out_dict = {index: out_dict[index] for index in sorted_out_dict_key}

    # Output Path
    raw_json_file_path = (
        project_root_path / "data" / "interim" / "raw_segmentation.json"
    )

    # Save file to data/interim/raw_plist_json.json
    with raw_json_file_path.open(mode="w") as file:
        file.write(json.dumps(sorted_out_dict, separators=(",", ":")))


def clean_raw_segmentation_json(
    project_root_path: pathlib.Path, json_path: pathlib.Path
) -> dict:
    """
    Pipeline function for cleaning raw segmentation json

    Args:
        project_root_path: path to root project
        json_path: path to raw segmentation json

    Returns:

    """
    # Open json file
    with json_path.open(mode="r") as json_file:
        raw_json_dict = json.load(json_file)

    clean_output_dict = clean_raw_segmentation_dict(project_root_path, raw_json_dict)

    # Output Path
    clean_json_file_path = (
        project_root_path / "data" / "interim" / "clean_segmentation.json"
    )

    # Save file to data/interim/raw_plist_json.json
    with clean_json_file_path.open(mode="w") as file:
        file.write(json.dumps(clean_output_dict, separators=(",", ":")))


def get_binary_segmentation_json(
    project_root_path: pathlib.Path, cleaned_json_path: pathlib.Path
):
    """
    This is a pipeline function for extracting binary segmentation
    out of the cleaned data

    Args:
        project_root_path:
        cleaned_json_path:
    """
    with cleaned_json_path.open(mode="r") as json_file:
        clean_json_dict = json.load(json_file)
    binary_segmentation_dict = split_clean_segmentation_to_binary(clean_json_dict)

    binary_segmentation_json_path = (
        project_root_path / "data" / "interim" / "binary_segmentation.json"
    )

    with binary_segmentation_json_path.open(mode="w") as json_file:
        json_file.write(json.dumps(binary_segmentation_dict, separators=(",", ":")))


def get_multiclass_segmentation_json(
    project_root_path: pathlib.Path, cleaned_json_path: pathlib.Path
):
    """
    This is a pipeline function for extracting multiclass segmentation
    out of the cleaned data

    Args:
        project_root_path:
        cleaned_json_path:
    """
    with cleaned_json_path.open(mode="r") as json_file:
        clean_json_dict = json.load(json_file)
    multiclass_segmentation_dict = split_clean_segmentation_to_multiclass(
        clean_json_dict
    )

    multiclass_segmentation_json_path = (
        project_root_path / "data" / "interim" / "multiclass_segmentation.json"
    )

    with multiclass_segmentation_json_path.open(mode="w") as json_file:
        json_file.write(json.dumps(multiclass_segmentation_dict, separators=(",", ":")))


def preprocess_segmentation_pipeline():
    """A function to run all preprocessing pipeline"""
    project_root_path = pathlib.Path.cwd()
    raw_json_file_path = (
        project_root_path / "data" / "interim" / "raw_segmentation.json"
    )

    clean_json_file_path = (
        project_root_path / "data" / "interim" / "clean_segmentation.json"
    )

    # Preprocess Segmentation
    # Convert all plist segmentation file into a json file
    create_raw_segmentation_json(project_root_path)
    clean_raw_segmentation_json(project_root_path, raw_json_file_path)
    get_binary_segmentation_json(project_root_path, clean_json_file_path)
    get_multiclass_segmentation_json(project_root_path, clean_json_file_path)


if __name__ == "__main__":
    preprocess_segmentation_pipeline()
