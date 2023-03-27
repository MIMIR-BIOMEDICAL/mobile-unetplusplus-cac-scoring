"""Preprocessing lib segmentation Test"""

import pathlib

from src.data.preprocess.lib.segmentation import (
    clean_raw_segmentation_dict,
    convert_plist_to_dict,
    split_clean_segmentation_to_binary,
    split_clean_segmentation_to_multiclass,
)


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
                            "Name": "Left Anterior Descending Artery",
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

    no_roi_test_dict = {
        "000": {
            "Images": [
                {
                    "ImageIndex": 1,
                    "NumberOfROIs": 1,
                    "ROIs": [
                        {
                            "Area": 0,
                            "Center": "(0, 0, 0)",
                            "Dev": 0,
                            "IndexInImage": 0,
                            "Length": 0,
                            "Max": 0,
                            "Mean": 0,
                            "Min": 0,
                            "Name": "Left Anterior Descending Artery",
                            "NumberOfPoints": 0,
                            "Point_mm": [],
                            "Point_px": [],
                            "Total": 0,
                            "Type": 0,
                        }
                    ],
                }
            ]
        }
    }
    cleaned_dict = clean_raw_segmentation_dict(test_dict)
    cleaned_no_roi_dict = clean_raw_segmentation_dict(no_roi_test_dict)

    assert cleaned_dict == {
        "000": [{"idx": "001", "roi": [{"loc": "LAD", "pos": [(1, 1)]}]}]
    }

    assert cleaned_no_roi_dict == {"000": []}


def test_split_clean_segmentation_to_binary():
    """Test function for binary segmentation extraction"""
    cleaned_dict = {
        "000": [
            {
                "idx": "001",
                "roi": [
                    {"loc": "LAD", "pos": [(1, 1)]},
                    {"loc": "LAD", "pos": [(3, 2)]},
                ],
            },
            {"idx": "002", "roi": [{"loc": "LAD", "pos": [(2, 2)]}]},
        ]
    }
    binary_segmentation_dict = split_clean_segmentation_to_binary(cleaned_dict)

    assert binary_segmentation_dict == {
        "000": [
            {"idx": "001", "pos": [(1, 1), (3, 2)]},
            {"idx": "002", "pos": [(2, 2)]},
        ]
    }


def test_split_clean_segmentation_to_multiclass():
    """Test function for multiclass segmentation extraction"""
    cleaned_dict = {
        "000": [
            {
                "idx": "001",
                "roi": [
                    {"loc": "LAD", "pos": [(1, 1)]},
                    {"loc": "LAD", "pos": [(3, 2)]},
                ],
            },
            {"idx": "002", "roi": [{"loc": "LAD", "pos": [(2, 2)]}]},
        ]
    }
    multiclass_segmentation_dict = split_clean_segmentation_to_multiclass(cleaned_dict)

    assert multiclass_segmentation_dict == {
        "000": [
            {
                "idx": "001",
                "roi": [
                    {"loc": 1, "pos": [(1, 1)]},
                    {"loc": 1, "pos": [(3, 2)]},
                ],
            },
            {"idx": "002", "roi": [{"loc": 1, "pos": [(2, 2)]}]},
        ]
    }
