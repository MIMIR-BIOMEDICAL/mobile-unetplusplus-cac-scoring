"""Module containing utility function for preprocessing"""
import numpy as np


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


def artery_loc_to_abbr(string_input: str):
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


def convert_num_to_abr(input_num: int) -> str:
    """
    This function convert artery location number
    encoding to its abbreviation


    Args:
        input_num: artery location encoding

    Returns:

    """
    conversion_dict = {
        1: "LAD",
        2: "RCA",
        3: "LCX",
        4: "LCA",
    }

    return conversion_dict.get(input_num, 0)


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


def filtered_patient_number_zfill_range(min_val: int, max_val: int):
    """
    A function that make a list within the range of
    min_val and max_val with an added filter from blacklist

    Args:
        min_val:
        max_val:

    Returns:

    """
    patient_number_list = patient_number_zfill_range(min_val, max_val)

    set_patient_number = set(tuple(patient_number_list))

    set_pixel_overlap = set(tuple(blacklist_pixel_overlap()))
    set_mislabelled_roi = set(tuple(blacklist_mislabelled_roi()))
    set_multiple_image = set(tuple(blacklist_multiple_image_id()))
    set_invalid_dicom = set(tuple(blacklist_invalid_dicom()))
    set_no_image = set(tuple(blacklist_no_image()))

    set_patient_number = (
        set_patient_number
        - set_pixel_overlap
        - set_mislabelled_roi
        - set_multiple_image
        - set_invalid_dicom
        - set_no_image
    )

    return sorted(list(set_patient_number))


def train_test_val_split(input_list, split, random_seed=811):
    """
    A function that split a list into 3 part base
    on the train,test, and val split inside the split

    Args:
        input_list ():
        split ():
        random_seed ():

    Returns:

    """
    data = input_list
    np.random.seed(random_seed)
    np.random.shuffle(data)

    num_samples = len(data)
    num_train = int(num_samples * split[0])
    num_val = int(num_samples * split[1])

    train_data = data[:num_train]
    val_data = data[num_train : num_train + num_val]
    test_data = data[num_train + num_val :]

    return {
        "train": train_data,
        "val": val_data,
        "test": test_data,
    }


def get_pos_from_bin_list(bin_list, idx):
    """
    A function that get the segmentation pos
    from a list of binary roi

    Args:
        bin_list ():
        idx ():

    Returns:

    """
    filtered_list = filter(lambda x: x["idx"] == idx, bin_list)
    pos = list(filtered_list)[0]["pos"]
    return pos


def get_pos_from_mult_list(mult_list, idx):
    """
    A function that get the segmentation pos
    from a list of multiclass roi

    Args:
        mult_list ():
        idx ():

    Returns:

    """
    filtered_list = filter(lambda x: x["idx"] == idx, mult_list)
    rois = list(filtered_list)[0]["roi"]

    out_list = []
    for roi in rois:
        loc = roi["loc"]
        for pos in roi["pos"]:
            out_list.append([pos[0], pos[1], loc])
    return out_list


def get_patient_split(split_arr: list, random_seed=811):
    """
    A function that create the randomize patient base on
    train,test, val split and also a random_seed

    Args:
        random_seed ():
        split_arr:

    Returns:

    """
    if len(split_arr) != 3:
        raise Exception("Split array should have only 3 member")

    if sum(split_arr) > 1:
        raise Exception("Split array should have the sum equaling to 1")

    calc_patient_arr = filtered_patient_number_zfill_range(0, 450)
    no_calc_patient_arr = filtered_patient_number_zfill_range(451, 789)

    calc_split = train_test_val_split(calc_patient_arr, split_arr, random_seed)
    no_calc_split = train_test_val_split(no_calc_patient_arr, split_arr, random_seed)

    calc_split["test"].extend(no_calc_split["test"])

    return calc_split


def split_list(lst, n):
    """
    Divides a list into a list of n sublists of approximately equal length.

    Args:
        lst (list): The list to split.
        n (int): The number of sublists to create.

    Returns:
        A list of n sublists.
    """
    # Calculate the number of elements per sublist
    quotient, remainder = divmod(len(lst), n)
    sizes = [quotient + 1 if i < remainder else quotient for i in range(n)]

    # Use list slicing to create sublists
    sublists = []
    start = 0
    for size in sizes:
        sublists.append(lst[start : start + size])
        start += size

    return sublists
