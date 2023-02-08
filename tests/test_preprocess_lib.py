"""Preprocessing Function Test"""

import pathlib

from src.data.preprocess.lib.segmentation import convert_plist_to_dict


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
