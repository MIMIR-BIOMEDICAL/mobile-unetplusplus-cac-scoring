"""Utils for automatic acac scoring system"""
import pathlib
import sys

import numpy as np

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
