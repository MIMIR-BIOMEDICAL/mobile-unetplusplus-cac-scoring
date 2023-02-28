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


def blacklist_pixel_overlap():
    """
    A function that return the list of patients with
    overlapping pixel between roi after float to int
    conversion

    The reason for this blacklist is due to the fact
    that Agatston Score need the correct surface area

    Returns:

    """
    return [
        "132",
        "428",
        "004",
        "037",
        "116",
        "144",
        "300",
        "161",
        "283",
        "303",
        "154",
        "305",
        "289",
        "387",
        "013",
    ]


def blacklist_mislabelled_roi():
    """
    A function that return the list of patients
    with mislabelled roi (No artery location)



    Returns:

    """
    return ["398", "238"]


def blacklist_multiple_image_id_with_roi():
    """
    A function that return the list of patients
    with multiple image and has roi label

    Returns:

    """
    return [
        "192",
        "276",
        "435",
        "155",
        "189",
        "358",
        "194",
        "228",
        "078",
        "417",
        "165",
        "146",
        "120",
        "156",
    ]


def blacklist_multiple_image_id():
    """
    A function that return the list of patients
    with multiple idx in their image data

    Returns:

    """
    return [
        "607",
        "641",
        "358",
        "194",
        "417",
        "453",
        "493",
        "156",
        "726",
        "155",
        "685",
        "228",
        "684",
        "146",
        "192",
        "189",
        "165",
        "700",
        "078",
        "276",
        "435",
        "398",
        "638",
        "513",
        "545",
        "741",
        "120",
    ]


def blacklist_invalid_dicom():
    """
    A function that return the list of patients
    with invalid dicom data

    Returns:

    """
    return ["159"]


def blacklist_no_image():
    """
    A function that return the list of patients
    that have missing image entirely

    Returns:

    """
    return ["012", "197", "598"]


def patient_number_zfill_range(min_val: int, max_val: int) -> list:
    """
    A function that make a list within the range of
    min_val and max_val to be use as patient number

    Args:
        min_val: minimum range value
        max_val: maximum range value

    Returns:

    """
    output = []
    for i in range(min_val, max_val + 1):
        output.append(str(i).zfill(3))
    return output
