from mmdet.apis import init_detector, inference_detector
import mmcv
import torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = 100000000000


def detector(file , checkpoint, device = "cpu"):

    """Create detector object - a realisation of the model supplied to the function via the relevant config and checkpoint files.

    Args:
        file (str): Location of config file.
        checkpoint (str): Location of checkpoint file.
        device (str, optional): Device to use for inference - CPU or GPU. Defaults to "cpu".

    Returns:
        model: MMDetect model object generated.
    """

    # Specify the path to model config and checkpoint file
    config_file = file
    checkpoint_file = checkpoint

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device=torch.device(device))
    return model