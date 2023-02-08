"""Module for preprocessing segmentation file"""
import pathlib
import plistlib


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
