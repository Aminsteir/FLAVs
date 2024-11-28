import torch

def circular_loss(predictions, targets):
    """
    Circular loss function to penalize angular differences.

    Args:
        predictions (torch.Tensor): Predicted [sin, cos] values.
        targets (torch.Tensor): Ground truth [sin, cos] values.

    Returns:
        torch.Tensor: Loss value.
    """
    sin_pred, cos_pred = predictions[:, 0], predictions[:, 1]
    sin_target, cos_target = targets[:, 0], targets[:, 1]
    return torch.mean(1 - (sin_pred * sin_target + cos_pred * cos_target))