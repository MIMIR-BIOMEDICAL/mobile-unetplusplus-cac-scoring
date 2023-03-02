"""Module containing the pipeline for preprocessing image file"""
import math
import pathlib
import sys

import click
import h5py
import pydicom as pdc
from tqdm import tqdm

sys.path.append(pathlib.Path.cwd().as_posix())
from src.data.preprocess.lib.image import \
    extract_patient_dicom_path  # pylint: disable=import-error,wrong-import-position
from src.data.preprocess.lib.utils import \
    patient_number_zfill_range  # pylint: disable=import-error,wrong-import-position


def extract_patient_dicom_data_to_h5(
    project_root_path: pathlib.Path, batch_step: int, sample: bool
):  # pylint: disable=too-many-locals
    """
    Pipeline function that extract dicom data (patient number,index, image, HU),
    batch and save them in a h5 file

    Args:
        project_root_path:
        batch_step:
    """
    gated_path = list(project_root_path.parent.rglob("Gated_release_final"))[0]

    sorted_patient_image_dict = extract_patient_dicom_path(gated_path)
    max_patient_num = int(list(sorted_patient_image_dict.keys())[-1])
    total_batch = math.ceil(max_patient_num / batch_step)

    h5_index_file_path = (
        project_root_path / "data" / "interim" / "image_h5" / "index.h5"
    )
    for batch_num in tqdm(
        range(total_batch), desc="Processing batch", unit="patient batch"
    ):
        patient_num_in_batch = patient_number_zfill_range(
            (batch_step + 1) * batch_num, batch_step * (batch_num + 1) + batch_num
        )

        h5_file_path = (
            project_root_path
            / "data"
            / "interim"
            / "image_h5"
            / f"dicom-image-patient-{patient_num_in_batch[0]}-{patient_num_in_batch[-1]}.h5"
        )
        # Create folder if not already exist
        h5_file_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(
            h5_file_path,
            "w",
        ) as file:
            for patient_num in tqdm(
                patient_num_in_batch,
                total=batch_step,
                desc="Processing patient",
                unit="patient",
                leave=False,
            ):
                patient_img_list = sorted_patient_image_dict.get(patient_num, None)

                # Skip if patient num not in patient_image_dict
                if patient_img_list is None:
                    continue

                patient_num_group = file.create_group(f"{patient_num}")
                patient_root_image_group = patient_num_group.create_group("img")

                sample_data = pdc.dcmread(patient_img_list[0]["path"])
                patient_num_group.create_dataset(
                    "pxl_spc", data=sample_data.PixelSpacing
                )
                patient_num_group.create_dataset(
                    "slc_thc", data=sample_data.SliceThickness
                )

                with h5py.File(h5_index_file_path, "a") as index_file:
                    index_file[f"/{patient_num}"] = h5py.ExternalLink(
                        h5_file_path, f"/{patient_num}"
                    )

                for patient_img in tqdm(
                    patient_img_list, desc="Processing Image", unit="img", leave=False
                ):
                    img_dcm = pdc.dcmread(patient_img["path"])
                    img_array = img_dcm.pixel_array
                    img_hu = pdc.pixel_data_handlers.util.apply_modality_lut(
                        img_array, img_dcm
                    )

                    patient_image_group = patient_root_image_group.create_group(
                        f"{patient_img['idx']}"
                    )

                    # Image dataset
                    patient_image_group.create_dataset(
                        "img_arr",
                        data=img_array,
                        compression="gzip",
                        compression_opts=9,
                        chunks=True,
                    )
                    # HU dataset
                    patient_image_group.create_dataset(
                        "img_hu",
                        data=img_hu,
                        compression="gzip",
                        compression_opts=9,
                        chunks=True,
                    )
        if sample:
            break


@click.command()
@click.option("-b", "--batch", type=click.INT, help="Patient Batch Size", default=10)
@click.option("-s", "--sample", type=click.BOOL, help="Sample Mode", default=False)
def preprocess_image_pipeline(batch, sample):
    """
    Cli interface for image pipeline

    Args:
        batch ():
    """
    project_root_path = pathlib.Path.cwd()
    extract_patient_dicom_data_to_h5(project_root_path, batch_step=batch, sample=sample)


if __name__ == "__main__":
    preprocess_image_pipeline()  # pylint: disable=no-value-for-parameter
