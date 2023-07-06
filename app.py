import pathlib
import sys
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pydicom as pdc
import streamlit as st
import tensorflow as tf
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

sys.path.append(pathlib.Path.cwd().as_posix())
from src.data.preprocess.lib.utils import get_patient_split
from src.models.lib.builder import build_unet_pp
from src.models.lib.config import UNetPPConfig
from src.models.lib.data_loader import preprocess_img
from src.models.lib.loss import dice_coef_nosq, log_cosh_dice_loss
from src.models.lib.utils import loss_dict_gen
from src.system.pipeline.output import auto_cac, ground_truth_auto_cac


@st.cache_resource
def load_main_model(project_root_path):
    model_root_path = project_root_path / "models" / "basic"  # Change this
    model_paths = next((model_root_path).rglob("*model*"))

    selected_model_path = model_paths.as_posix()
    loss_func = log_cosh_dice_loss

    main_model = tf.keras.models.load_model(
        selected_model_path,
        custom_objects={
            "log_cosh_dice_loss": loss_func,
            "dice_coef_nosq": dice_coef_nosq,
        },
    )

    return main_model


@st.cache_resource
def load_model(project_root_path, depth, _main_model):
    loss_func = log_cosh_dice_loss
    model_depth = 5
    filter_list = [16, 32, 64, 128, 256]

    pruned_model = {}

    pruned_model[f"d{depth}"] = {}

    model_config = UNetPPConfig(
        model_name=f"model_d{depth}",
        upsample_mode="transpose",
        depth=depth + 1,
        input_dim=[512, 512, 1],
        batch_norm=True,
        deep_supervision=False,
        model_mode="basic",
        n_class={"bin": 1},
        filter_list=filter_list[: depth + 1],
    )

    model, output_layer_name = build_unet_pp(model_config, custom=True)

    print(f"-- Creating pruned model d{depth}")
    for layer in tqdm(model.layers):
        pruned_layer_name = layer.name

        main_model_layer = _main_model.get_layer(pruned_layer_name)

        main_model_weight = main_model_layer.get_weights()

        layer.set_weights(main_model_weight)

    pruned_model[f"d{depth}"]["model"] = model

    loss_dict = loss_dict_gen(model_config, output_layer_name, [loss_func])

    pruned_model[f"d{depth}"]["config"] = model_config
    pruned_model[f"d{depth}"]["loss_dict"] = loss_dict

    metrics = ["acc"]

    pruned_model[f"d{depth}"]["model"].compile(
        optimizer=tf.keras.optimizers.legacy.Adam(),
        loss=pruned_model[f"d{depth}"]["loss_dict"],
        metrics=metrics,
    )

    return pruned_model


def plot(container, img, title):
    fig, ax = plt.subplots()
    ax.set_title(title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.imshow(img, cmap="gray")
    ax.axis("off")
    fig.colorbar(im, cax=cax, orientation="vertical")
    container.pyplot(fig)


def plot_lesion(container, img, bin_mask, title):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.imshow(img, cmap="gray")
    ax.imshow(bin_mask, cmap="gray", alpha=0.5)
    ax.axis("off")
    container.pyplot(fig)


def main():
    project_root_path = pathlib.Path.cwd()
    main_model = load_main_model(project_root_path)
    st.header(
        "AUTOMATED SCORING OF CORONARY ARTERY CALCIUM FROM CT-SCAN IMAGES USING CNN FOR HEART DISEASE RISK ASSESSMENT"
    )
    st.subheader("Aditya Wardianto 07311940000001")
    c1, c2 = st.columns([1, 5])
    c1.image(
        "https://www.its.ac.id/wp-content/uploads/2020/07/Lambang-ITS-2-300x300.png",
    )
    file = c2.file_uploader("Upload DICOM file here", accept_multiple_files=True)
    temp_dir = tempfile.TemporaryDirectory()

    if file:
        # Breakdown mode
        if len(file) == 1:
            uploaded_file_name = "temp.dcm"
            uploaded_file_path = pathlib.Path(temp_dir.name) / uploaded_file_name
            with open(uploaded_file_path, "wb") as out_temp_file:
                out_temp_file.write(file[0].read())

            st.subheader("Breakdown Mode")
            with st.form("breakdown_form"):
                ds = st.selectbox(
                    label="Deep Supervision Layer", options=["DS1", "DS2", "DS3", "DS4"]
                )
                if st.form_submit_button():
                    ds_dict = {"DS1": 1, "DS2": 2, "DS3": 3, "DS4": 4}
                    num = ds_dict[ds]
                    pruned_model = load_model(project_root_path, num, main_model)
                    output = auto_cac(
                        [uploaded_file_path], pruned_model[f"d{num}"]["model"]
                    )
                    a, b, c = st.columns(3)
                    d, e, f = st.columns(3)
                    plot(a, output["slice"][0]["img_arr"], "Input Image")
                    plot(b, output["slice"][0]["img_hu"], "HU Image")
                    plot(c, output["slice"][0]["img_clip"], "Clipped Image")
                    plot(d, output["slice"][0]["img_norm"], "Normalized Image")
                    plot(e, output["slice"][0]["img_zero"], "Zero Centered Image")
                    plot_lesion(
                        f,
                        output["slice"][0]["img_hu"],
                        output["slice"][0]["pred_bin"],
                        "Binary Segmentation Mask",
                    )
                    st.write(
                        f"Agatston Score for the current slice: {output['total_agatston']}"
                    )
        else:
            file_path_list = []
            file_dict = {}
            with st.spinner("Creating temporary dicom filej"):
                for idx, f in enumerate(file):
                    uploaded_file_name = f"temp{idx}.dcm"
                    uploaded_file_path = (
                        pathlib.Path(temp_dir.name) / uploaded_file_name
                    )
                    with open(uploaded_file_path, "wb") as out_temp_file:
                        out_temp_file.write(f.read())
                    file_path_list.append(uploaded_file_path)
                    file_dict[uploaded_file_path] = f.name

            # Whole Scan mode
            st.subheader("Whole Scan Mode")
            with st.form("whole_scan"):
                ds = st.selectbox(
                    label="Deep Supervision Layer", options=["DS1", "DS2", "DS3", "DS4"]
                )
                if st.form_submit_button():
                    ds_dict = {"DS1": 1, "DS2": 2, "DS3": 3, "DS4": 4}
                    num = ds_dict[ds]
                    with st.spinner("Creating Pruned Model"):
                        pruned_model = load_model(project_root_path, num, main_model)
                    with st.spinner("Model Predicting"):
                        output = auto_cac(
                            file_path_list,
                            pruned_model[f"d{num}"]["model"],
                            mem_opt=True,
                        )
                    st.write(
                        f"Total Agatston score for all slices: {output['total_agatston']}"
                    )
                    st.write(f"Stratified risk: {output['class']}")
                    if output["total_agatston"]!=0:
                        st.write("Calcium Detected on the following file:")
                        real_path_list = []
                        for tmp_path in output["detected"]:
                            real_path_list.append(file_dict[tmp_path])
                        st.write(real_path_list)


if __name__ == "__main__":
    main()
