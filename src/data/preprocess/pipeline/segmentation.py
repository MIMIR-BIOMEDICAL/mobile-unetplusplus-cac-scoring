"""Module containing the pipeline for preprocessing segmentation file"""
import json
import pathlib
import sys

from tqdm import tqdm

sys.path.append(pathlib.Path.cwd().as_posix())

from src.data.preprocess.lib.segmentation import \
    convert_plist_to_dict  # pylint: disable=import-error,wrong-import-position


def create_raw_segmentation_json(project_root_path: pathlib.Path):
    """
    A part of pipeline function that convert all of segmentation
    plist file to json. No prior preprocessing is done

    Args:
        project_root_path: Path to the main working directory
    """

    # Find xml all plist file inside data
    xml_file_list = project_root_path.rglob("*.xml")

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
        file.write(json.dumps(sorted_out_dict))
