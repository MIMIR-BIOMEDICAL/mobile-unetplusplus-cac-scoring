"""Utils for automatic acac scoring system"""
import pathlib
import sys

import numpy as np

from src.data.preprocess.lib.utils import convert_num_to_abr

sys.path.append(pathlib.Path.cwd().as_posix())


def ccl(img):
    """
    Performs Connected Component Labeling on a binary image.

    Args:
        img (ndarray): Binary input image.

    Returns:
        tuple: A tuple containing the number of connected components and the labeled image.

    """

    h, w = img.shape

    # Pad the image with a border of 1
    padded_img = np.pad(img, 1, mode="constant").astype(int)

    label = 1
    hash_map = {}

    for row in range(1, h + 1):
        for col in range(1, w + 1):
            if padded_img[row, col] == 1:
                neighbours = np.array(
                    [
                        padded_img[row, col - 1],
                        padded_img[row - 1, col - 1],
                        padded_img[row - 1, col],
                        padded_img[row - 1, col + 1],
                    ]
                )

                non_zero_indices = np.nonzero(neighbours)[0]
                non_zero_count = len(non_zero_indices)

                if non_zero_count == 0:
                    padded_img[row, col] = label
                    label += 1

                elif non_zero_count == 1:
                    padded_img[row, col] = neighbours[non_zero_indices[0]]
                else:
                    min_label = np.min(neighbours[non_zero_indices])
                    padded_img[row, col] = min_label

                    for neighbour in neighbours[non_zero_indices]:
                        if neighbour != min_label:
                            hash_map.setdefault(
                                min_label, []).append(neighbour)
                            hash_map[min_label] = np.unique(
                                hash_map[min_label]
                            ).tolist()

    new_hash_map = {}
    visited = set()

    def dfs(key):
        visited.add(key)
        values = hash_map.get(key, [])
        for value in values:
            if value not in visited:
                dfs(value)
                if value in hash_map:
                    values.extend(hash_map[value])
                    del hash_map[value]

    for key in list(hash_map.keys()):
        if key not in visited:
            dfs(key)
            new_hash_map[key] = list(set(hash_map[key]))

    for key, values in new_hash_map.items():
        for val in values:
            padded_img = np.where(padded_img == val, key, padded_img)

    count = len(np.unique(padded_img))

    return count, padded_img[1:-1, 1:-1]


def get_lesion_dict(lesion_labels, lesion_stats):
    """
    Create a dictionary of lesion information from the labeled image and statistics.

    Args:
        lesion_labels (numpy.ndarray): Labeled image obtained from connected component labeling.
        lesion_stats (numpy.ndarray): Statistics obtained from connected component labeling.

    Returns:
        dict: Dictionary containing lesion information with lesion index as keys.
              Each entry in the dictionary has 'loc' and 'stats' attributes.
              'loc' contains the coordinates of the lesion pixels.
              'stats' contains the statistics of the lesion.

    Note:
        This function is designed to work with the output of cv2.connectedComponentsWithStats() function in OpenCV.
    """
    lesion_dict = {}
    for index, stat in enumerate(lesion_stats):
        # Skip the background (index 0)
        if index == 0:
            continue
        lesion_dict[index] = {
            "loc": np.argwhere(lesion_labels == index),
            "stats": stat,
        }

    return lesion_dict


def assign_lesion_type(prediction, lesion_dict):
    """
    Assign lesion types based on the prediction values to the lesion dictionary.

    Args:
        prediction (numpy.ndarray): 1D array containing the predicted values for lesion pixels.
        lesion_dict (dict): Dictionary containing lesion information.

    Returns:
        dict: Updated lesion dictionary with assigned lesion types.

    Note:
        The prediction array should have the same length as the number of lesion pixels in the lesion dictionary.

    """
    for index, lesion in lesion_dict.items():
        lesion_prediction = prediction[tuple(zip(*lesion["loc"]))]
        unique, count = np.unique(lesion_prediction, return_counts=True)
        lesion_dict[index]["type"] = unique[np.argmax(count)]
    return lesion_dict


def agatston(image_hu, lesion_dict, spacing_pair):
    """
    Calculate Agatston scores for lesions based on Hounsfield Unit (HU) values.

    Args:
        image_hu (numpy.ndarray): 2D array of Hounsfield Unit (HU) values.
        lesion_dict (dict): Dictionary containing lesion information.
        spacing_pair (tuple): Pair of values representing pixel spacing in millimeters.

    Returns:
        dict: Dictionary containing Agatston scores for each lesion type, as well as the total Agatston score.

    Note:
        The image_hu should be in raw format, without any preprocessing applied.

    """
    agatston_dict = {}
    for lesion in lesion_dict.values():
        # find max attenuation
        max_att = np.max(image_hu[tuple(zip(*lesion["loc"]))])

        # get weight
        if max_att < 130:
            w = 0
        elif 130 <= max_att < 200:
            w = 1
        elif 200 <= max_att < 300:
            w = 2
        elif 300 <= max_att < 400:
            w = 3
        else:
            w = 4

        # find area
        square_area = spacing_pair[0] * spacing_pair[1]
        area = square_area * lesion["loc"].shape[0]

        score = area * w
        abbr = convert_num_to_abr(lesion["type"])
        agatston_dict[abbr] = agatston_dict.get(abbr, 0) + score

    agatston_dict["total"] = np.sum(list(agatston_dict.values()))
    return agatston_dict
