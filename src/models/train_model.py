"""Module for containing training cli"""
import json
import os
import pathlib
import sys
from datetime import datetime

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import inquirer
import numpy as np
import tensorflow as tf
from keras.utils.layer_utils import count_params
from tensorflow import keras

sys.path.append(pathlib.Path.cwd().as_posix())
from src.models.lib.builder import build_unet_pp
from src.models.lib.config import UNetPPConfig
from src.models.lib.data_loader import create_dataset
from src.models.lib.loss import (categorical_focal_loss, dice_coef_func,
                                 dice_loss_func, log_cosh_dice_loss,
                                 weighted_categorical_crossentropy)
from src.models.lib.utils import loss_dict_gen, parse_list_string


def train_model(
    project_root_path,
    metadata: dict,
    model_config: UNetPPConfig,
    custom: bool,
    batch_size: int,
    shuffle_size: int,
    epochs: int,
    loss_function_list: list,
    learning_rate,
):
    """
    Train a model using the specified configuration.

    Parameters:
    -----------
    project_root_path : str
        The root path of the project.
    model_config : UNetPPConfig
        The configuration object for the model.
    custom : str
        The custom object for the model.
    batch_size : int
        The batch size for training.
    shuffle_size : int
        The size for shuffle buffer.
    epochs : int
        The number of epochs to train for.
    lost_function_list : list, optional
        The list of loss functions to use, defaults to [log_cosh_loss, "categorical_crossentropy"].
    metrics : list, optional
        The list of metrics to use, defaults to ["acc"].

    Returns:
    --------
    None
    """
    print("[1] Preparing Model...")

    devices = tf.config.experimental.list_physical_devices("GPU")

    if len(devices) > 1:
        print("Using Multi GPU")
        devices_name = [d.name.split("e:")[1] for d in devices]
        strategy = tf.distribute.MirroredStrategy(devices_name)
        with strategy.scope():
            metrics = [
                dice_coef_func(use_bg=False),
                tf.keras.metrics.OneHotMeanIoU(num_classes=5),
                tf.keras.metrics.Recall(),
                tf.keras.metrics.Precision(),
            ]
            model, model_layer_name = build_unet_pp(model_config, custom=custom)

            loss_dict = loss_dict_gen(
                model_config,
                model_layer_name,
                loss_function_list,
            )

            model.compile(
                optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate),
                loss=loss_dict,
                metrics=metrics,
            )
    else:
        metrics = [
            dice_coef_func(use_bg=False),
            tf.keras.metrics.OneHotMeanIoU(num_classes=5),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.Precision(),
        ]
        model, model_layer_name = build_unet_pp(model_config, custom=custom)

        loss_dict = loss_dict_gen(
            model_config,
            model_layer_name,
            loss_function_list,
        )

        model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate),
            loss=loss_dict,
            metrics=metrics,
        )

    # Create model folder and save metadata
    model_folder = project_root_path / "models" / metadata.get("model_name")
    model_folder.mkdir(parents=True, exist_ok=True)

    metadata["trainable_weights"] = count_params(model.trainable_weights)
    metadata["non_trainable_weights"] = count_params(model.non_trainable_weights)
    metadata["weights"] = count_params(model.weights)

    (model_folder / "metadata.txt").write_text(json.dumps(metadata, indent=4))

    per_epoch_path = f"models/{model_config.model_name}/" + "model-epoch-{epoch:02d}.h5"
    best_model_path = (
        f"models/{model_config.model_name}/" + "best-model-epoch-{epoch:02d}.h5"
    )

    epoch_callback = keras.callbacks.ModelCheckpoint(
        per_epoch_path,
        monitor="val_loss",
        verbose=0,
        save_best_only=False,
        save_weights_only=False,
        mode="min",
    )
    best_callback = keras.callbacks.ModelCheckpoint(
        best_model_path,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode="min",
    )
    history_callback = keras.callbacks.CSVLogger(
        f"models/{model_config.model_name}/history.csv"
    )

    print("--- Model Prepared")

    # Create Dataset
    print("[2] Loading Dataset...")
    coca_dataset = create_dataset(
        project_root_path, model_config, batch_size, shuffle_size
    )

    train_coca_dataset = coca_dataset["train"]
    val_coca_dataset = coca_dataset["val"]
    test_coca_dataset = coca_dataset["test"]

    print("--- Dataset Loaded")
    try:
        print("[3] Start Model Training...")
        model.fit(
            x=train_coca_dataset,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=val_coca_dataset,
            callbacks=[history_callback, best_callback, epoch_callback],
        )
        print("--- Training Finished")
        print("--- Saving Latest Model...")
        model.save(f"models/{model_config.model_name}/model_epoch_latest.h5")
        print("--- Latest Model Saved")
        print("--- [4] Evaluate on test dataset")
        model.evaluate(test_coca_dataset)
    except KeyboardInterrupt:
        print("--- Training Interrupted")
        print("--- Saving Latest Model...")
        model.save(f"models/{model_config.model_name}/model_epoch_latest_interupted.h5")
        print("--- Latest Model Saved")
        print("--- [4] Evaluate on test dataset")
        model.evaluate(test_coca_dataset)


def start_prompt():
    """
    This function prompts the user with a series of questions to configure the model.

    Returns:
        answer (dict): A dictionary containing the user's answers to the configuration questions.

    """

    q = [
        inquirer.Text("model_name", message="Model Name"),
        inquirer.List(
            "model_mode",
            message="Model Mode",
            choices=["basic", "mobile", "sanity_check"],
            default="mobile",
        ),
        inquirer.Confirm(
            "use_default_config", message="Use default config?", default=True
        ),
        inquirer.List(
            "upsample_mode",
            message="Upsample Mode",
            choices=["upsample", "transpose"],
            default="transpose",
            ignore=lambda x: x["use_default_config"],
        ),
        inquirer.Text(
            "depth",
            message="Model Depth",
            default="5",
            ignore=lambda x: x["use_default_config"],
        ),
        inquirer.Text(
            "input_dim",
            message="Input Dimension (x,y,z)",
            default="512,512,1",
            ignore=lambda x: x["use_default_config"],
        ),
        inquirer.Confirm(
            "batch_norm",
            message="Batch Norm",
            default=True,
            ignore=lambda x: x["use_default_config"],
        ),
        inquirer.Confirm(
            "deep_supervision",
            message="Deep Supervision",
            default=True,
            ignore=lambda x: x["use_default_config"],
        ),
        inquirer.Text(
            "filter_list",
            message="Number of Filter per layer",
            default="16,32,64,128,256",
            ignore=lambda x: x["use_default_config"],
        ),
        inquirer.Text(
            "downsample_iteration",
            message="Number of downsample iteration per layer",
            default="1,2,3,3,2",
            ignore=lambda x: x["use_default_config"] or x["model_mode"] == "basic",
        ),
        inquirer.Text("batch_size", message="Batch Size", default="32"),
        inquirer.Text("shuffle_size", message="Shuffle Size", default="64"),
        inquirer.Text("epochs", message="Epochs", default="10000"),
        inquirer.List(
            "loss_func",
            message="Loss Function",
            choices=[
                "Focal",
                "Dice",
                "Log Cosh Dice",
                "Weighted Categorical Crossentropy",
            ],
            default="Focal",
        ),
        inquirer.Text(
            "alpha",
            message="Focal Loss Alpha",
            default="0.25",
            ignore=lambda x: x["loss_func"] != "Focal",
        ),
        inquirer.Text(
            "gamma",
            message="Focal Loss Gamma",
            default="2",
            ignore=lambda x: x["loss_func"] != "Focal",
        ),
        inquirer.Text("learning_rate", message="Learning Rate", default="0.001"),
    ]

    try:
        answer = inquirer.prompt(q)
    except KeyboardInterrupt:
        sys.exit(1)

    return answer


def prompt_parser(answer) -> dict:
    """
    Parses and modifies the user's answers from the prompt.

    Args:
        answer (dict): A dictionary containing the user's answers from the prompt.

    Returns:
        parsed_answer (dict): A dictionary containing the modified and parsed answers.

    """
    answer["model_name"] = f"{answer['model_name']}-{datetime.now().isoformat()}"
    answer["depth"] = int(answer["depth"])
    answer["input_dim"] = parse_list_string(answer["input_dim"])
    answer["filter_list"] = parse_list_string(answer["filter_list"])

    # If model_mode is "basic", set downsample_iteration to None
    answer["downsample_iteration"] = (
        None
        if answer["model_mode"] == "basic"
        else parse_list_string(answer["downsample_iteration"])
    )

    answer["batch_size"] = int(answer["batch_size"])
    answer["shuffle_size"] = int(answer["shuffle_size"])
    answer["epochs"] = int(answer["epochs"])
    answer["epochs"] = int(answer["epochs"])
    answer["learning_rate"] = float(answer["learning_rate"])
    answer["alpha"] = float(answer["alpha"])
    answer["gamma"] = float(answer["gamma"])

    return answer


def main():
    """
    Main function for executing the model training workflow.

    Returns:
        None

    """

    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    except ValueError:
        tpu = None

    if tpu:
        policy = "mixed_bfloat16"
    else:
        policy = "mixed_float16"
    tf.keras.mixed_precision.set_global_policy(policy)

    project_root_path = pathlib.Path.cwd()

    answer = start_prompt()

    parsed_answer = prompt_parser(answer)

    if parsed_answer["loss_func"] == "Focal":
        loss_func = [
            categorical_focal_loss(
                alpha=parsed_answer.get("alpha"), gamma=parsed_answer.get("gamma")
            )
        ]
    elif parsed_answer["loss_func"] == "Dice":
        loss_func = [dice_loss_func]
    elif parsed_answer["loss_func"] == "Log Cosh Dice":
        loss_func = [log_cosh_dice_loss]
    elif parsed_answer["loss_func"] == "Weighted Categorical Crossentropy":
        loss_func = [
            weighted_categorical_crossentropy(
                [
                    0.010000347481757162,
                    782.61385178081,
                    760.2813517781574,
                    1412.843535626247,
                    5752.665350699007,
                ]
            )
        ]
    # Create model configuration
    if answer.get("model_mode") == "sanity_check":
        config = UNetPPConfig(
            model_name=parsed_answer.get("model_name"),
            upsample_mode="upsample",
            depth=3,
            input_dim=[512, 512, 1],
            batch_norm=True,
            model_mode="mobile",
            n_class={"mult": 5},
            deep_supervision=True,
            filter_list=[2, 2, 2],
            downsample_iteration=[1, 1, 1],
        )
        train_model(
            project_root_path,
            parsed_answer,
            config,
            custom=True,
            batch_size=parsed_answer.get("batch_size"),
            shuffle_size=parsed_answer.get("shuffle_size"),
            epochs=parsed_answer.get("epochs"),
            loss_function_list=loss_func,
            learning_rate=parsed_answer.get("learning_rate"),
        )
        return

    config = UNetPPConfig(
        model_name=parsed_answer.get("model_name"),
        upsample_mode=parsed_answer.get("upsample", "upsample"),
        depth=parsed_answer.get("depth", 2),
        input_dim=parsed_answer.get("input_dim", [1, 1, 1]),
        batch_norm=parsed_answer.get("batch_norm", True),
        model_mode=parsed_answer.get("model_mode"),
        n_class={"mult": 5},
        deep_supervision=parsed_answer.get("deep_supervision", True),
        filter_list=parsed_answer.get("filter_list", [1, 1]),
        downsample_iteration=parsed_answer.get("downsample_iteration", [1, 1]),
    )

    # Train the model
    if answer.get("use_default_config"):
        train_model(
            project_root_path,
            parsed_answer,
            config,
            custom=False,
            batch_size=parsed_answer.get("batch_size"),
            shuffle_size=parsed_answer.get("shuffle_size"),
            epochs=parsed_answer.get("epochs"),
            loss_function_list=loss_func,
            learning_rate=parsed_answer.get("learning_rate"),
        )
        return
    else:
        train_model(
            project_root_path,
            parsed_answer,
            config,
            custom=True,
            batch_size=parsed_answer.get("batch_size"),
            shuffle_size=parsed_answer.get("shuffle_size"),
            epochs=parsed_answer.get("epochs"),
            loss_function_list=loss_func,
            learning_rate=parsed_answer.get("learning_rate"),
        )


if __name__ == "__main__":
    main()
