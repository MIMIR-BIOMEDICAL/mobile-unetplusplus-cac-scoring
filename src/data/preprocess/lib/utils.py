"""Module containing utility function for preprocessing"""


def string_to_int_tuple(string_input: str) -> list:
    """
    This function convert a string input with the format "( 1xxxx.xxxx, 1xxxx.xxxx)"
    into a list

    Args:
        string_input: string formatted in "( 1xxxx.xxxx, 1xxxx.xxxx)"

    Returns:
        out_list: list of integer position

    """
    string_input = string_input.strip("()")
    string_input_list = string_input.split(",")
    out_list = []

    for string_num in string_input_list:
        out_list.append(int(float(string_num)))

    return tuple(out_list)


def artery_loc_to_abbr(string_input: str) -> str | None:
    """
    Convert string artery location into its abbreviation, will
    return None if not found in conversion dict

    Args:
        string_input: string for full artery location

    Returns:
        A string abbreivation of the artery locatoin


    """
    if string_input not in [
        "Left Anterior Descending Artery",
        "Right Coronary Artery",
        "Left Circumflex Artery",
        "Left Coronary Artery",
    ]:
        return None

    conversion_dict = {
        "Left Anterior Descending Artery": "LAD",
        "Right Coronary Artery": "RCA",
        "Left Circumflex Artery": "LCX",
        "Left Coronary Artery": "LCA",
    }

    return conversion_dict[string_input]


def find_duplicates(lists_of_lists: list) -> list:
    """
    This function find duplicate inside a list of list

    Args:
        lists_of_lists (): list of list containing

    Returns:

    """
    seen = set()
    duplicates = []
    for lst in lists_of_lists:
        lst = tuple(lst)
        if lst in seen:
            duplicates.append(lst)
        else:
            seen.add(lst)
    return duplicates


def convert_abr_to_num(input_string: str) -> int:
    """
    This function convert artery location abbreviation
    to its number encoding


    Args:
        input_string: artery location abbreviation

    Returns:

    """
    conversion_dict = {
        "LAD": 1,
        "RCA": 2,
        "LCX": 3,
        "LCA": 4,
    }

    return conversion_dict.get(input_string, 0)
