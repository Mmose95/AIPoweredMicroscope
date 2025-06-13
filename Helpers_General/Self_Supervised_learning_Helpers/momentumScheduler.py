import math

def get_dinov2_teacher_momentum(epoch: int, max_epoch: int, base_momentum: float = 0.996) -> float:
    """
    Sine-based momentum schedule for DINOv2 teacher update.

    Args:
        epoch (int): Current epoch number.
        max_epoch (int): Total number of training epochs.
        base_momentum (float): Starting EMA momentum.

    Returns:
        float: Scheduled momentum for the current epoch.
    """
    return 1 - (1 - base_momentum) * (math.cos(math.pi * epoch / max_epoch) + 1) / 2