"""Dataclass to configure UNet++ Model"""
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class UNetPPConfig:
    """
    UNetPPConfig: A dataclass for configuring a UNet++ model.

     Attributes:
         model_name (str): The model name
         input_dim (List[int]): a list of integers representing the input dimensions of the UNet++ model.
         depth (int): the depth of the UNet++ model, which corresponds to the number of times the feature map size is halved.
         n_class (Dict[str, int]): a dictionary containing the number of classes for binary and/or multi-class segmentation tasks. The dictionary must have a string key and an integer value greater than or equal to 1.
         deep_supervision (bool): a flag indicating whether or not to use deep supervision.
         batch_norm (bool): a flag indicating whether or not to use batch normalization.
         upsample_mode (str): the upsample mode for the model. Can be either "upsample" or "transpose".
         filter_list (List[int]): a list of integers representing the number of filters in each convolutional block of the UNet++ model.
         downsample_iteration (Optional[List[int]]): an optional list of integers used in Mobile UNet++ representing the number of times iteration in each layer.

     Methods:
         validate_depth(value): a method to validate the depth attribute. Raises a ValueError if the depth value is less than or equal to 0.
         validate_n_class(value): a method to validate the n_class attribute. Raises a ValueError if the dictionary does not have a "bin" or "mult" key or if the value associated with each key is less than 1.
         validate_upsample_mode(value): a method to validate the upsample_mode attribute. Raises a ValueError if the upsample_mode value is not "upsample" or "transpose".
         validate_filter_list(value): a method to validate the filter_list attribute. Raises a ValueError if the length of the filter_list is not equal to the depth or if any value in the list is less than 1.
         validate_downsample_iteration(value): a method to validate the downsample_iteration attribute. Raises a ValueError if the length of the downsample_iteration list is not equal to the depth or if any value in the list is less than 1.

    """

    model_name: str
    input_dim: List[int]
    depth: int
    n_class: Dict[str, int]
    deep_supervision: bool
    batch_norm: bool
    upsample_mode: str
    model_mode: str
    filter_list: List[int]
    downsample_iteration: Optional[List[int]] = None

    def __setattr__(self, prop, val):
        """Overrides the __setattr__ method to validate the input.

        Args:
            prop (str): The name of the attribute.
            val: The value to be set to the attribute.

        Raises:
            ValueError: If the input fails any of the validation checks.

        Returns:
            None
        """

        if validator := getattr(self, f"validate_{prop}", None):
            object.__setattr__(self, prop, validator(val) or val)
        else:
            super().__setattr__(prop, val)

    def validate_model_name(self, value):
        """Validates the model_name attribute.

        Args:
            value (str): The value to be validated.


        Returns:
            None
        """
        if not isinstance(value, str):
            raise ValueError("model_name need to be string")

    def validate_input_dim(self, value):
        """Validates the input_dim attribute.

        Args:
            value (List[int]): The value to be validated.

        Raises:
            ValueError: If the input value is less than or equal to 0.

        Returns:
            None
        """
        if len(value) != 3:
            raise ValueError("input_dim need 3 value inside its list")
        if not all(isinstance(val, int) and val >= 1 for val in value):
            raise ValueError(
                "input_dim values must be integers greater than or equal to 1"
            )

    def validate_depth(self, value):
        """Validates the depth attribute.

        Args:
            value (int): The value to be validated.

        Raises:
            ValueError: If the input value is less than or equal to 0.

        Returns:
            None
        """
        if value <= 0:
            raise ValueError("depth need to be >=1")

    def validate_n_class(self, value):
        """Validates the n_class attribute.

        Args:
            value (Dict[str, int]): The value to be validated.

        Raises:
            ValueError: If the dictionary doesn't contain either 'bin' or 'mult' as keys,
                        or if any of the values are not integers greater than or equal to 1.

        Returns:
            Dict[str, int]: The validated input value.
        """

        if not all(isinstance(val, int) and val >= 1 for val in value.values()):
            raise ValueError(
                "n_class dictionary values must be integers greater than or equal to 1"
            )
        return value

    def validate_model_mode(self, value):
        """Validates the model_mode attribute.

        Args:
            value (str): The value to be validated.

        Raises:
            ValueError: If the input value is not 'basic' or 'mobile'.

        Returns:
            None
        """

        if value not in ["basic", "mobile"]:
            raise ValueError("model_mode can only be either 'basic' or 'mobile'")

    def validate_upsample_mode(self, value):
        """Validates the upsample_mode attribute.

        Args:
            value (str): The value to be validated.

        Raises:
            ValueError: If the input value is not 'upsample' or 'transpose'.

        Returns:
            None
        """

        if value not in ["upsample", "transpose"]:
            raise ValueError(
                "upsample_mode can only be either 'upsample' or 'transpose'"
            )

    def validate_filter_list(self, value):
        """Validates the filter_list attribute.

        Args:
            value (List[int]): The value to be validated.

        Raises:
            ValueError: If the list length is not equal to depth,
                        or if any of the values are not integers greater than or equal to 1.

        Returns:
            None
        """

        if len(value) != self.depth:
            raise ValueError("List of filter length need to be equal to depth")
        if not all(isinstance(val, int) and val >= 1 for val in value):
            raise ValueError(
                "filter_list values must be integers greater than or equal to 1"
            )

    def validate_downsample_iteration(self, value):
        """Validates the downsample_iteration attribute.

        Args:
            value (Optional[List[int]]): The value to be validated.

        Raises:
            ValueError: If the list length is not equal to depth,
                        or if any of the values are not integers greater than or equal to 1.

        Returns:
            None
        """

        if value is not None:
            if len(value) != self.depth:
                raise ValueError(
                    "List of downsample_iteration need to be equal to depth"
                )
            if not all(isinstance(val, int) and val >= 1 for val in value):
                raise ValueError(
                    "downsample_iteration values must be integers greater than or equal to 1"
                )
