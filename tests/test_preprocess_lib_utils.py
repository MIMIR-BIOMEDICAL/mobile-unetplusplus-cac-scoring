"""Preprocessing lib utils test"""
from src.data.preprocess.lib.utils import (artery_loc_to_abbr,
                                           convert_abr_to_num,
                                           string_to_int_tuple)


def test_string_to_int_tuple():
    """Test converting a string of tuple float into a list of int"""
    test_string = "(1.000, 1.000)"
    out_list = string_to_int_tuple(test_string)
    assert out_list == (1, 1)


def test_convert_artery_location_to_abbreviation():
    test_string = "Left Anterior Descending Artery"
    fail_test_string = "1"

    test_output = artery_loc_to_abbr(test_string)
    fail_test_output = artery_loc_to_abbr(fail_test_string)

    assert test_output == "LAD"
    assert fail_test_output is None


def test_convert_abr_to_num():
    test_string = "LAD"
    test_num = convert_abr_to_num(test_string)
    assert test_num == 1

    fail_test_num = convert_abr_to_num("ABC")
    assert fail_test_num == 0
