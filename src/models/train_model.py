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
from tensorflow import keras

sys.path.append(pathlib.Path.cwd().as_posix())
from src.models.lib.builder import build_unet_pp
from src.models.lib.config import UNetPPConfig
from src.models.lib.data_loader import create_dataset
from src.models.lib.loss import log_cosh_dice_loss_func
from src.models.lib.utils import loss_dict_gen, parse_list_string


class SaveBestModel(keras.callbacks.Callback):
    """
    Keras callback that saves the best model based on the average validation loss.

    Args:
        config (UNetPPConfig): Configuration object for the UNet++ model.

    Attributes:
        config (UNetPPConfig): Configuration object for the UNet++ model.
        best_loss (float): The best average validation loss seen so far.
        filepath (str): The path to save the best model to.

    Methods:
        on_epoch_end(epoch, logs): Keras callback that saves the model if the average validation loss
            is better than the current best loss.
    """

    def __init__(self, config: UNetPPConfig):
        super().__init__()
        self.config = config
        self.best_loss = float("inf")
        self.filepath = f"models/{config.model_name}"

    def on_epoch_end(self, epoch, logs=None):
        """
        Keras callback that saves the model if the average validation loss is better than the current best loss.

        Args:
            epoch (int): The current epoch number.
            logs (dict): Dictionary containing the training and validation loss values.

        Returns:
            None
        """
        loss_keys = [key for key in logs.keys() if "val" in key]

        loss_values = [logs[key] for key in loss_keys]

        avg_loss = np.mean(loss_values)

        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            filename = f"{self.filepath}/model_epoch_{str(epoch).zfill(7)}.h5"
            self.model.save(filename)
            print(f"Saved model at {filename} with avg_loss {avg_loss:.4f}")


def train_model(
    project_root_path,
    model_config: UNetPPConfig,
    custom: bool,
    batch_size: int,
    shuffle_size: int,
    epochs: int,
    lost_function_list: list = [log_cosh_dice_loss_func, "categorical_crossentropy"],
    metrics: list = ["acc"],
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
    print("Creating Model...")
    # Create model
    model, model_layer_name = build_unet_pp(model_config, custom=custom)
    print("Model Created")

    # Create Dataset
    print("Loading Dataset...")
    coca_dataset = create_dataset(
        project_root_path, model_config, model_layer_name, batch_size, shuffle_size
    )

    train_coca_dataset = coca_dataset["train"]
    val_coca_dataset = coca_dataset["val"]

    print("Dataset Loaded")
    # Model Compilation and Training

    print("Preparing Model for Training...")
    loss_dict = loss_dict_gen(
        model_config,
        model_layer_name,
        lost_function_list,
    )

    model.compile(optimizer="adam", loss=loss_dict, metrics=metrics)

    model_callback = SaveBestModel(model_config)
    history_callback = keras.callbacks.CSVLogger(
        f"models/{model_config.model_name}/history.csv"
    )

    print("Model Prepared")
    try:
        print("Start Model Training...")
        model.fit(
            x=train_coca_dataset,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=val_coca_dataset,
            callbacks=[model_callback, history_callback],
        )
        print("Training Finished")
        print("Saving Latest Model...")
        model.save(f"models/{model_config.model_name}/model_epoch_latest.h5")
        print("Latest Model Saved")
    except KeyboardInterrupt:
        print("Training Interrupted")
        print("Saving Latest Model...")
        model.save(f"models/{model_config.model_name}/model_epoch_latest_interupted.h5")
        print("Latest Model Saved")


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
            choices=["basic", "mobile"],
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
        inquirer.Text("epochs", message="epochs", default="10000"),
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

    # Create model folder and save metadata
    model_folder = project_root_path / "models" / parsed_answer.get("model_name")
    model_folder.mkdir(parents=True, exist_ok=True)
    (model_folder / "metadata.txt").write_text(json.dumps(parsed_answer, indent=4))

    # Create model configuration
    config = UNetPPConfig(
        model_name=parsed_answer.get("model_name"),
        upsample_mode=parsed_answer.get("upsample", "upsample"),
        depth=parsed_answer.get("depth", 2),
        input_dim=parsed_answer.get("input_dim", [1, 1, 1]),
        batch_norm=parsed_answer.get("batch_norm", True),
        model_mode=parsed_answer.get("model_mode"),
        n_class={"bin": 1, "mult": 5},
        deep_supervision=parsed_answer.get("deep_supervision", True),
        filter_list=parsed_answer.get("filter_list", [1, 1]),
        downsample_iteration=parsed_answer.get("downsample_iteration", [1, 1]),
    )

    # Train the model
    if answer.get("use_default_config"):
        train_model(
            project_root_path,
            config,
            custom=False,
            batch_size=parsed_answer.get("batch_size"),
            shuffle_size=parsed_answer.get("shuffle_size"),
            epochs=parsed_answer.get("epochs"),
        )
    else:
        train_model(
            project_root_path,
            config,
            custom=True,
            batch_size=parsed_answer.get("batch_size"),
            shuffle_size=parsed_answer.get("shuffle_size"),
            epochs=parsed_answer.get("epochs"),
        )


if __name__ == "__main__":
    main()
