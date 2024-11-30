import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, output_type="angle"):
        """
        Base model to handle dynamic output types.

        Args:
            output_type (str): The format of the output (e.g., "angle", "angle_norm", "sin_cos").
        """
        super(BaseModel, self).__init__()
        self.output_type = output_type  # Set the output type

    def format_output(self, output):
        """
        Format the model's output based on the specified output type.

        Args:
            output (torch.Tensor): The raw output from the model.

        Returns:
            torch.Tensor: Formatted output.
        """
        if self.output_type == "angle":
            return output.squeeze(-1)  # Single angle in degrees
        elif self.output_type == "angle_norm":
            return torch.tanh(output.squeeze(-1))  # Map to [-1, 1]
        elif self.output_type == "sin_cos":
            norm = output / torch.linalg.norm(output, ord=2, dim=1, keepdim=True)  # Normalize to [sin, cos]
            return norm
        else:
            raise ValueError(f"Unsupported output type: {self.output_type}")
