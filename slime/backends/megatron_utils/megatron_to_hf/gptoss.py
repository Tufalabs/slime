def convert_gptoss_to_hf(args, name, param):
    """
    Convert a single weight from Megatron format to HuggingFace format.

    Args:
        args: Megatron arguments containing model configuration
        name: Weight name in Megatron format
        param: Weight tensor

    Returns:
        list: List of tuples (hf_name, hf_param) for HuggingFace format
    """
    raise NotImplementedError()
