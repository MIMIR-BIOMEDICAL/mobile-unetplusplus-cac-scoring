"Module tfor system pipeline"
import pathlib
import sys

import cv2
import numpy as np
import pydicom as pdc

sys.path.append(pathlib.Path.cwd().as_posix())

from src.models.lib.data_loader import preprocess_img
from src.system.lib.utils import agatston, assign_lesion_type, ccl, get_lesion_dict


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

    return [count, label, stats, centroid]


def extract_dcm(img_dcm_path):
    img_dcm = pdc.dcmread(img_dcm_path)
    img_array = img_dcm.pixel_array
    img_hu = pdc.pixel_data_handlers.util.apply_modality_lut(img_array, img_dcm)
    pxl_spc = img_dcm.PixelSpacing

    return img_hu, pxl_spc


def auto_cac(img_dcm_paths, model):
    output_dict = {}

    for index, img_dcm_path in enumerate(img_dcm_paths):
        output_dict[index] = {}

        ## Preprocessing
        # Get Image HU and pixel spacing
        img_hu, pxl_spc = extract_dcm(img_dcm_path)
        output_dict[index]["img_hu"] = img_hu
        output_dict[index]["pxl_spc"] = pxl_spc

        # Prepare image to correct dims (1,N,N,1)
        expanded_img_batch = np.expand_dims(img_hu, axis=0)
        expanded_img_class = np.expand_dims(expanded_img_batch, axis=0)

        ## Model
        # Inference
        output_dict[index]["img_pred_one_hot"] = model.predict(expanded_img_class)

        ## Postprocessing
        # Reverse one-hot encoding
        img_pred_batchless = np.squeeze(output_dict[index]["img_pred_one_hot"], axis=0)
        output_dict[index]["img_pred"] = np.argmax(img_pred_batchless, axis=-1)

        # Connected Component
        connected_lesion = call_ccl(output_dict[index]["img_pred"], mode="cv2")
        lesion_dict = get_lesion_dict(connected_lesion)

        # Agatston scoring
        output_dict[index]["lesion"] = assign_lesion_type(
            output_dict[index]["img_pred"], lesion_dict
        )

        output_dict["agatston"] = agatston(
            output_dict[index]["img_pred"],
            output_dict[index]["lesion"],
            output_dict[index]["lesion"],
        )

    for values in output_dict.values():
        for key_name in ["total", "LAD", "RCA", "LCX", "LCA"]:
            output_dict[key_name] = (
                output_dict.get(key_name, 0) + values["agatston"][key_name]
            )
    return output_dict
