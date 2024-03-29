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
                            hash_map.setdefault(min_label, []).append(neighbour)
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


def get_lesion_dict(lesion_info):
    """
    Create a dictionary of lesion information from connected component labelling.

    Args:
        lesion_info (list): Output obtained from connected component labeling.

    Returns:
        dict: Dictionary containing lesion information with lesion index as keys.
              Each entry in the dictionary has 'loc' and 'stats' attributes.
              'loc' contains the coordinates of the lesion pixels.
              'stats' contains the statistics of the lesion.

    Note:
        This function is designed to work with the output of cv2.connectedComponentsWithStats() function in OpenCV.
    """
    lesion_dict = {}
    _, label, stats, _ = lesion_info
    for index, _ in enumerate(stats):
        # Skip the background (index 0)
        if index == 0:
            continue
        lesion_dict[index] = {
            "loc": np.argwhere(label == index),
        }

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
    agatston_score = 0
    for lesion in lesion_dict.values():
        # find max attenuation
        lesion_val = image_hu[tuple(zip(*lesion["loc"]))]
        max_att = np.max(lesion_val)
        # find area
        square_area = spacing_pair[0] * spacing_pair[1]
        area = square_area * lesion["loc"].shape[0]

        # get weight
        if max_att < 130 or area < 1:
            # No need to  count, if w =0
            continue
        if 130 <= max_att < 200:
            w = 1
        elif 200 <= max_att < 300:
            w = 2
        elif 300 <= max_att < 400:
            w = 3
        else:
            w = 4

        score = area * w
        agatston_score += int(score)
    return agatston_score
