"Module tfor system pipeline"
import pathlib
import sys

import cv2
import numpy as np
import pydicom as pdc

sys.path.append(pathlib.Path.cwd().as_posix())

from src.data.preprocess.lib.utils import convert_abr_to_num
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


def classify_risk(total_agatston):
    if total_agatston == 0:
        class_label = "Absent"
    elif 1 <= total_agatston <= 100:
        class_label = "Discrete"
    elif 101 <= total_agatston <= 400:
        class_label = "Moderate"
    elif total_agatston > 400:
        class_label = "Accentuated"
    else:
        class_label = None

    return class_label


def auto_cac(img_dcm_paths, model, mem_opt=False):
    output_dict = {}
    output_dict["slice"] = {}

    # Loop over  image path(s)
    for index, img_dcm_path in enumerate(img_dcm_paths):
        ## Preprocessing
        # Get Image HU and pixel spacing
        img_hu, pxl_spc = extract_dcm(img_dcm_path)

        # Prepare image to correct dims (1,N,N,1)
        preprocessed_img_hu = preprocess_img(img_hu)
        expanded_img_batch = np.expand_dims(preprocessed_img_hu, axis=0)
        expanded_img_class = np.expand_dims(expanded_img_batch, axis=3)

        ## Model
        # Inference
        pred_softmax = model.predict(expanded_img_class, verbose=0)

        ## Postprocessing
        # Reverse one-hot encoding
        pred_batchless = np.squeeze(pred_softmax[-1])
        pred_mult = np.argmax(pred_batchless, axis=-1)
        # Create binary mask to be use as ccl base
        pred_bin = np.where(pred_mult == 0, 0, 1)

        # Connected Component
        connected_lesion = call_ccl(pred_bin, mode="cv2")
        lesion_dict = get_lesion_dict(connected_lesion)
        # Use mult mask to assign lesion type
        lesion_dict = assign_lesion_type(pred_mult, lesion_dict)

        # Agatston scoring (per slice score)
        agatston_dict = agatston(img_hu, lesion_dict, pxl_spc)

        if not mem_opt:
            output_dict["slice"] = output_dict.get("slice", {})
            output_dict["slice"][index] = {}
            output_dict["slice"][index]["img_hu"] = img_hu
            output_dict["slice"][index]["pxl_spc"] = pxl_spc
            output_dict["slice"][index]["pred_softmax"] = pred_softmax
            output_dict["slice"][index]["pred_bin"] = pred_bin
            output_dict["slice"][index]["pred_mult"] = pred_mult
            output_dict["slice"][index]["lesion"] = lesion_dict
            output_dict["slice"][index]["agatston_slice_score_dict"] = agatston_dict

        for cor_art in ["LAD", "RCA", "LCX", "LCA"]:
            output_dict[f"{cor_art}_total_agatston"] = output_dict.get(
                f"{cor_art}_total_agatston", 0
            ) + agatston_dict.get(cor_art, 0)

        output_dict["total_agatston"] = (
            output_dict.get("total_agatston", 0) + agatston_dict["total"]
        )

    output_dict["class"] = classify_risk(output_dict["total_agatston"])

    return output_dict


def ground_truth_auto_cac(img_dcm_paths, roi_lists, mem_opt=False):
    output_dict = {}

    # Loop over  image path(s)
    for index, (img_dcm_path, roi_list) in enumerate(zip(img_dcm_paths, roi_lists)):
        ## Preprocessing
        # Get Image HU and pixel spacing
        img_hu, pxl_spc = extract_dcm(img_dcm_path)

        temp_mult = np.zeros((512, 512))
        for roi in roi_list:
            temp_mult[tuple(zip(*roi["pos"]))] = convert_abr_to_num(roi["loc"])

        temp_bin = np.where(temp_mult == 0, 0, 1)

        # Connected Component
        connected_lesion = call_ccl(temp_bin, mode="cv2")
        lesion_dict = get_lesion_dict(connected_lesion)
        lesion_dict = assign_lesion_type(temp_mult, lesion_dict)

        # Agatston scoring (per slice score)
        agatston_dict = agatston(img_hu, lesion_dict, pxl_spc)

        if not mem_opt:
            output_dict["slice"] = output_dict.get("slice", {})
            output_dict["slice"][index] = {}
            output_dict["slice"][index]["img_hu"] = img_hu
            output_dict["slice"][index]["pxl_spc"] = pxl_spc
            output_dict["slice"][index]["lesion"] = lesion_dict
            output_dict["slice"][index]["agatston_slice_score"] = agatston_dict

        for cor_art in ["LAD", "RCA", "LCX", "LCA"]:
            output_dict[f"{cor_art}_total_agatston"] = output_dict.get(
                f"{cor_art}_total_agatston", 0
            ) + agatston_dict.get(cor_art, 0)

        output_dict["total_agatston"] = (
            output_dict.get("total_agatston", 0) + agatston_dict["total"]
        )

    output_dict["class"] = classify_risk(output_dict["total_agatston"])
    return output_dict
