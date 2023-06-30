import json
import pathlib
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pydicom as pdc
import sklearn.metrics as skm
import tensorflow as tf
from keras.utils.layer_utils import count_params
from tqdm import tqdm

sys.path.append(pathlib.Path.cwd().parent.as_posix())

from src.data.preprocess.lib.utils import get_patient_split
from src.models.lib.builder import build_unet_pp
from src.models.lib.config import UNetPPConfig
from src.models.lib.data_loader import create_dataset, preprocess_img
from src.models.lib.loss import (dice_coef, dice_coef_nosq, log_cosh_dice_loss,
                                 log_cosh_dice_loss_nosq)
from src.models.lib.utils import loss_dict_gen
from src.system.pipeline.output import auto_cac, ground_truth_auto_cac

project_root_path = pathlib.Path.cwd().parent

# Select model
model_root_path = (
    project_root_path / "models" / "mend_bismillah_nodecay-2023-06-28_23:27"
)  # Change this
model_paths = list((model_root_path).rglob("*model-epoch-225*"))

# Import main model
selected_model_path = model_paths[0].as_posix()
loss_func = log_cosh_dice_loss

main_model = tf.keras.models.load_model(
    selected_model_path,
    custom_objects={
        "log_cosh_dice_loss": loss_func,
        "dice_coef_nosq": dice_coef_nosq,
    },
)

model_depth = 5
depth = int(sys.argv[1])
filter_list = [16, 32, 64, 128, 256]


pruned_model = {}


# for depth in range(1, model_depth):
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

    main_model_layer = main_model.get_layer(pruned_layer_name)

    main_model_weight = main_model_layer.get_weights()

    layer.set_weights(main_model_weight)

pruned_model[f"d{depth}"]["model"] = model

loss_dict = loss_dict_gen(model_config, output_layer_name, [loss_func])

# pruned_model[f"d{depth}"]["dataset"] = create_dataset(
#     project_root_path, model_config, 2, 1
# )
#
# pruned_model[f"d{depth}"]["config"] = model_config
pruned_model[f"d{depth}"]["loss_dict"] = loss_dict

# for depth in range(1, model_depth):
#     print(depth)
#
#     # pruned_model[f"d{depth}"]["trainable_weights"] = count_params(
#     #     pruned_model[f"d{depth}"]["model"].trainable_weights
#     # )
#     # pruned_model[f"d{depth}"]["non_trainable_weights"] = count_params(
#     #     pruned_model[f"d{depth}"]["model"].non_trainable_weights
#     # )
#     # pruned_model[f"d{depth}"]["weights"] = count_params(
#     #     pruned_model[f"d{depth}"]["model"].weights
#     # )

metrics = [
    "acc"
    # dice_coef,
    #         tf.keras.metrics.BinaryIoU(),
    # tf.keras.metrics.Recall(),
    # tf.keras.metrics.Precision(),
    #         tf.keras.metrics.TruePositives(),
    #         tf.keras.metrics.TrueNegatives(),
    #         tf.keras.metrics.FalseNegatives(),
    #         tf.keras.metrics.FalsePositives(),
]

pruned_model[f"d{depth}"]["model"].compile(
    optimizer=tf.keras.optimizers.legacy.Adam(),
    loss=pruned_model[f"d{depth}"]["loss_dict"],
    metrics=metrics,
)


patient_test_data = set(get_patient_split([0.7, 0.2, 0.1])["test"])

time_list = []

# Warmup
for idx_seg in tqdm(patient_test_data):
    patient_root_path = next(project_root_path.rglob(f"patient/{idx_seg.lstrip('0')}"))
    img_path = list(patient_root_path.rglob(f"*.dcm"))
    auto_cac(img_path, pruned_model[f"d{depth}"]["model"], mem_opt=True)

# Test
a = time.perf_counter()
for idx_seg in tqdm(patient_test_data):
    patient_root_path = next(project_root_path.rglob(f"patient/{idx_seg.lstrip('0')}"))
    img_path = list(patient_root_path.rglob(f"*.dcm"))
    auto_cac(img_path, pruned_model[f"d{depth}"]["model"], mem_opt=True)
    time_list.append(time.perf_counter() - a)

mean = np.mean(time_list)
std = np.std(time_list)
maks = np.max(time_list)
mins = np.min(time_list)

print(f"Model {depth} with {mean} plus minus {std} with max {maks} and min {mins}")
