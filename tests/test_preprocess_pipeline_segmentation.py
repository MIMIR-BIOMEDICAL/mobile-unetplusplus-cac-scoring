"""Preprocessing Function Test"""

import json
import pathlib

from src.data.preprocess.pipeline.segmentation import (
    clean_raw_segmentation_json,
    create_raw_segmentation_json,
    get_binary_segmentation_json,
)


def test_create_raw_segmentation_json(fs):  # pylint: disable=invalid-name
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

    fs.create_dir(cwd_path / "data" / "interim")

    create_raw_segmentation_json(cwd_path)

    json_path = cwd_path / "data" / "interim" / "raw_segmentation.json"

    with json_path.open(mode="r") as json_file:
        dict_output = json.load(json_file)

    assert dict_output == {
        "test": {
            "userId": 1,
            "id": 1,
            "title": "delectus aut autem",
            "completed": False,
        }
    }


def test_clean_raw_segmentation_json(fs):  # pylint: disable=invalid-name
    """Test pipeline for cleaning preprocess segmentation data"""
    cwd_path = pathlib.Path.cwd()
    test_data_path = cwd_path / "data" / "interim" / "raw_segmentation.json"

    fs.create_file(
        test_data_path,
        contents=r"""
        {
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
            "Point_mm": [
              "(1, 1, 1)"
            ],
            "Point_px": [
              "(1.00, 1.00)"
            ],
            "Total": 1,
            "Type": 1
          }
        ]
      }
    ]
  }
}
        """,
    )

    clean_raw_segmentation_json(cwd_path, test_data_path)

    json_path = cwd_path / "data" / "interim" / "clean_segmentation.json"

    with json_path.open(mode="r") as json_file:
        cleaned_dict_output = json.load(json_file)

    assert cleaned_dict_output == {
        "000": [{"idx": "001", "roi": [{"loc": "LAD", "pos": [[1, 1]]}]}]
    }

    test_data_path = cwd_path / "data" / "interim" / "fail_raw_segmentation.json"

    fs.create_file(
        test_data_path,
        contents=r"""
        {
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
            "Name": "1",
            "NumberOfPoints": 1,
            "Point_mm": [
              "(1, 1, 1)"
            ],
            "Point_px": [
              "(1.00, 1.00)"
            ],
            "Total": 1,
            "Type": 1
          }
        ]
      }
    ]
  }
}
        """,
    )

    clean_raw_segmentation_json(cwd_path, test_data_path)

    json_path = cwd_path / "data" / "interim" / "clean_segmentation.json"

    with json_path.open(mode="r") as json_file:
        cleaned_dict_output = json.load(json_file)

    assert cleaned_dict_output == {"000": []}


def test_binary_segmentation_json(fs):  # pylint: disable=invalid-name
    """Test pipeline for extracting binary segmentation data"""
    cwd_path = pathlib.Path.cwd()
    test_data_path = cwd_path / "data" / "interim" / "raw_segmentation.json"

    fs.create_file(
        test_data_path,
        contents=r"""
        {
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
            "Point_mm": [
              "(1, 1, 1)"
            ],
            "Point_px": [
              "(1.00, 1.00)"
            ],
            "Total": 1,
            "Type": 1
          }
        ]
      }
    ]
  }
}
        """,
    )

    clean_raw_segmentation_json(cwd_path, test_data_path)

    json_path = cwd_path / "data" / "interim" / "clean_segmentation.json"

    get_binary_segmentation_json(cwd_path, json_path)

    binary_json_path = cwd_path / "data" / "interim" / "binary_segmentation.json"

    with binary_json_path.open(mode="r") as file:
        binary_segmentation_dict = json.load(file)

    assert binary_segmentation_dict == {"000": [{"idx": "001", "pos": [[1, 1]]}]}

    test_data_path = cwd_path / "data" / "interim" / "fail_raw_segmentation.json"

    fs.create_file(
        test_data_path,
        contents=r"""
        {
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
            "Name": "1",
            "NumberOfPoints": 1,
            "Point_mm": [
              "(1, 1, 1)"
            ],
            "Point_px": [
              "(1.00, 1.00)"
            ],
            "Total": 1,
            "Type": 1
          }
        ]
      }
    ]
  }
}
        """,
    )

    clean_raw_segmentation_json(cwd_path, test_data_path)

    json_path = cwd_path / "data" / "interim" / "clean_segmentation.json"

    get_binary_segmentation_json(cwd_path, json_path)

    binary_json_path = cwd_path / "data" / "interim" / "binary_segmentation.json"

    with binary_json_path.open(mode="r") as file:
        binary_segmentation_dict = json.load(file)

    assert binary_segmentation_dict == {"000": []}
