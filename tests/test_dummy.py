"""A test dummy"""


def dummy(input_integer: int) -> int:
    """
    A dummy function that will be removed after ohter tests is added

    Args:
        input_integer (int): A dummy integer input

    Returns:
        output_integer (int): A dummy integer output
    """
    output_integer: int = input_integer + 1
    return output_integer


def test_dummy():
    """A test for the dummy function"""
    assert dummy(1) == 2
