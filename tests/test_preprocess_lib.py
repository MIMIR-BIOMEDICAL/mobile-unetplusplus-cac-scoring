"""Preprocessing Function Test"""

import pathlib

from src.data.preprocess.lib.segmentation import (clean_raw_segmentation_dict,
                                                  convert_plist_to_dict)
from src.data.preprocess.lib.utils import string_to_int_tuple


def test_convert_plist_to_dict(fs):  # pylint: disable=invalid-name
    """Test for convert_plist_to_dict function with mocked plist file"""
    cwd_path = pathlib.Path.cwd()
    test_data_path = (
        cwd_path
        / "data"
        / "raw"
        / "cocacoronarycalciumandchestcts-2"
        / "Gated_release_final"
        / "calcium_xml"
        / "test.xml"
    )

    fs.create_file(
        test_data_path,
        contents=r"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
	<dict>
		<key>userId</key>
		<integer>1</integer>
		<key>id</key>
		<integer>1</integer>
		<key>title</key>
		<string>delectus aut autem</string>
		<key>completed</key>
		<false/>
	</dict>
</plist>""",
    )

    dict_output = convert_plist_to_dict(test_data_path)
    assert dict_output == {
        "userId": 1,
        "id": 1,
        "title": "delectus aut autem",
        "completed": False,
    }


def test_clean_raw_segmentation_dict():
    """Test for clean_raw_segmentation_dict"""
    test_dict = {
        "000": {
            "Images": [
                {
                    "ImageIndex": 1,
                    "NumberOfROIs": 1,
                    "ROIs": [
                        {
                            "Area": 1,
                            "Center": "(1, 1, 1)",
                            "Dev": 1,
                            "IndexInImage": 0,
                            "Length": 1,
                            "Max": 1,
                            "Mean": 1,
                            "Min": 1,
                            "Name": "Another Breaking Changes Done",
                            "NumberOfPoints": 1,
                            "Point_mm": ["(1, 1, 1)"],
                            "Point_px": ["(1.00, 1.00)"],
                            "Total": 1,
                            "Type": 1,
                        }
                    ],
                }
            ]
        }
    }

    cleaned_dict = clean_raw_segmentation_dict(test_dict)

    assert cleaned_dict == {
        "000": [{"idx": 1, "roi": [{"name": "ABC", "pos": [(1, 1)]}]}]
    }


def test_string_to_int_tuple():
    """Test converting a string of tuple float into a list of int"""
    test_string = "(1.000, 1.000)"
    out_list = string_to_int_tuple(test_string)
    assert out_list == (1, 1)
