"""Builder module as interface to building model"""
import pathlib
import sys

from tensorflow import keras  # pylint: disable=wrong-import-position,import-error

sys.path.append(pathlib.Path.cwd().parent.as_posix())
from src.models.lib.base import base_unet_pp, unetpp_mobile_backend
from src.models.lib.config import UNetPPConfig


def build_unet_pp(config: UNetPPConfig, custom: bool = False) -> keras.Model:
    """
    Builds a UNet++ model.

    Args:
        config: Configuration object specifying the parameters of the model.
        custom: Whether to build a custom model or use the default parameters.

    Returns:
        A UNet++ model and a list containing the output head name.

    Raises:
        ValueError: If the specified configuration is invalid.
    """
    if custom:
        if config.model_mode == "basic":
            if config.downsample_iteration is not None:
                raise ValueError(
                    "You are using basic mode, downsample_iteration is not required"
                )
        elif config.model_mode == "mobile":
            if config.downsample_iteration is None:
                raise ValueError(
                    "You are using mobile mode, downsample_iteration is required"
                )
        else:
            raise ValueError(f"Invalid model mode: {config.model_mode}")

        return unetpp_mobile_backend(config)

    if config.model_mode == "basic":
        model_conf = UNetPPConfig(
            model_name=config.model_name,
            input_dim=[512, 512, 1],
            batch_norm=True,
            model_mode="basic",
            depth=5,
            n_class={"bin": 1},
            deep_supervision=True,
            upsample_mode="transpose",
            filter_list=[32, 64, 128, 256, 512],
        )

    elif config.model_mode == "mobile":
        model_conf = UNetPPConfig(
            model_name=config.model_name,
            upsample_mode="transpose",
            depth=5,
            input_dim=[512, 512, 1],
            batch_norm=True,
            model_mode="mobile",
            n_class={"bin": 1},
            deep_supervision=True,
            filter_list=[16, 32, 64, 128, 256],
            downsample_iteration=[4, 3, 2, 2, 1],
        )
    else:
        raise ValueError(f"Invalid model mode: {config.model_mode}")

    # return base_unet_pp(model_conf)
    return unetpp_mobile_backend(model_conf)
