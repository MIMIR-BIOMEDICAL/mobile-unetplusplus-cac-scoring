"""Preprocessing Function Test"""

import json
import pathlib

from src.data.preprocess.pipeline.segmentation import \
    create_raw_segmentation_json


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
