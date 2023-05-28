"Module tfor system pipeline"
import pathlib
import sys

import cv2
import numpy as np

sys.path.append(pathlib.Path.cwd().as_posix())

from src.system.lib.utils import ccl


def call_ccl(img, mode="cv2"):
    """
    Performs Connected Component Labeling on a binary image using either OpenCV's function or a basic implementation.

    Args:
        img (ndarray): Binary input image.
        mode (str, optional): The mode to use for CCL. Can be "cv2" to use OpenCV's function or "basic" for a basic implementation. Defaults to "cv2".

    Returns:
        tuple: A tuple containing the number of connected components, the labeled image, component statistics, and component centroids (if available).

    """

    count = 0
    label = []
    stats = []
    centroid = []

    if mode == "cv2":
        count, label, stats, centroid = cv2.connectedComponentsWithStats(
            np.uint8(img), connectivity=8
        )
    elif mode == "basic":
        count, label = ccl(img)

    return count, label, stats, centroid
