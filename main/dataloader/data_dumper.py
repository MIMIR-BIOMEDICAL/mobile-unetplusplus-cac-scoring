import os
import pickle
import sys

import pydicom
from tqdm import tqdm


def get_image_from_dicom(path: str, output_byte: bool = False):
    """Extract image data from DICOM file, returning data according to output_byte parameter.

    Args:
        path (str): Path to DICOM file.
        output_byte (bool, optional): Change output type from numpy array to bytes. Defaults to False.

    Returns:
        data: Image data from DICOM, can be represented in numpy array or bytes.
    """
    data = pydicom.dcmread(path).pixel_array
    if output_byte:
        return data.tobytes
    else:
        return data


def get_coca_dicom_dict_path(path: str):
    """Get dictionary of DICOM paths for each patient. This function only accept the path for raw unedited COCA Dataset folder.

    Args:
        path (string): Path to COCA Dataset folder.

    Returns:
        patient_dcm_path (dict): Dictionary with key as patient ID and value as list of DICOM paths.
    """
    patient_folders = os.listdir(path)
    patient_dcm_path = {}

    for patient_folder in patient_folders:
        full_patient_folder_path = os.path.join(path, patient_folder)
        folder_in_patient_folder = os.listdir(full_patient_folder_path)

        dcm_path = os.path.join(full_patient_folder_path, folder_in_patient_folder[0])
        full_dcm_path = [os.path.join(dcm_path, path) for path in os.listdir(dcm_path)]

        patient_dcm_path[patient_folder] = full_dcm_path

    return patient_dcm_path


def _check_folder(folder_name: str):
    """Check if folder exists, if not, create it.

    Args:
        folder_name (str): Folder Name
    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def _split_dict(dict: dict, split_size: int):
    pass


def main():
    # Dump COCA Dataset into TFRecord files
    args = len(sys.argv)

    if args == 1:
        raise ValueError("Please provide the path to COCA Dataset folder.")
    elif args == 2:
        path = sys.argv[1]

    _check_folder(path)

    dcm_path_dict = get_coca_dicom_dict_path(
        r"data/cocacoronarycalciumandchestcts-2/Gated_release_final/patient/"
    )


if __name__ == "__main__":
    main()
