"""Data Preprocessor"""

import pathlib
import plistlib


def convert_plist_to_dict(path: pathlib.Path) -> dict:
    """
    This function convert a plist file (in xml form) to dictionary

    Args:
        path (pathlib.Path): Path to file

    Returns:
        output (dict): A dictionary representation of the input file
    """
    # Check if file have xml extension
    if path.suffix != ".xml":
        raise Exception("File neeeded to be using xml extension")

    # Check if file exists
    if not path.exists():
        raise Exception("File doesn't exist")

    # Get dictionary out of plist file
    with path.open(mode="rb") as file:
        output = plistlib.load(file)

    return output
