import numpy as np
import torch
from models.registry import get_model
from utils.loss import circular_loss


class ModelConfig:
    def __init__(self, model_type, output_type, loss_function=None):
        """
        Configuration for the model, dataset, and training process.

        Args:
            model_type (str): Type of model (e.g., "dual_stream", "temporal_transformer").
            output_type (str): Output type (e.g., "angle", "angle_norm", "sin_cos").
            loss_function (callable, optional): Custom loss function for training.
        """
        self.model_type = model_type
        self.output_type = output_type
        self.loss_function = loss_function or self._default_loss_function()

    def _default_loss_function(self):
        """
        Get the default loss function based on the output type.

        Returns:
            callable: Loss function.
        """
        if self.output_type == "sin_cos":
            return circular_loss
        elif self.output_type in ["angle", "angle_norm"]:
            return torch.nn.MSELoss()
        else:
            raise ValueError(f"Unsupported output type: {self.output_type}")

    def get_model(self, **kwargs):
        """
        Retrieve and configure the model based on the model type.

        Args:
            **kwargs: Additional parameters for model initialization.

        Returns:
            nn.Module: Configured model.
        """
        return get_model(self.model_type, output_type=self.output_type, **kwargs)

    def prepare_targets(self, angle):
        """
        Convert steering angle to the target format based on the output type.

        Args:
            angle (float): Steering angle in degrees.

        Returns:
            torch.Tensor: Processed target.
        """
        if self.output_type == "angle":
            return torch.tensor(angle, dtype=torch.float32)
        elif self.output_type == "angle_norm":
            return torch.tensor(angle / 180, dtype=torch.float32)  # Normalize to [-1, 1]
        elif self.output_type == "sin_cos":
            angle_rad = np.deg2rad(angle)
            return torch.tensor([np.sin(angle_rad), np.cos(angle_rad)], dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported output type: {self.output_type}")

    def convert_output_to_angle(self, output):
        """
        Convert the model's output back to an angle for visualization or evaluation.

        Args:
            output (torch.Tensor): Model output.

        Returns:
            float: Predicted steering angle in degrees.
        """
        if self.output_type == "angle":
            return output.item()  # Already in degrees
        elif self.output_type == "angle_norm":
            return output.item() * 180  # Convert from [-1, 1] to degrees
        elif self.output_type == "sin_cos":
            sin, cos = output[0].item(), output[1].item()
            return np.rad2deg(np.arctan2(sin, cos))  # Convert from sin_cos to degrees
        else:
            raise ValueError(f"Unsupported output type: {self.output_type}")
