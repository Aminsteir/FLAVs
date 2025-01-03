import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatioTemporalModel(nn.Module):
    def __init__(self, image_size=(256, 455), **kwargs):
        super(SpatioTemporalModel, self).__init__()
        self.spatial_temporal_conv = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2))
        )

        # Dynamically compute flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 3, image_size[0], image_size[1])  # [batch, channels, depth, height, width]
            conv_output = self.spatial_temporal_conv(dummy_input)
            self.flattened_size = conv_output.view(-1).size(0)

        self.fc = nn.Sequential(
            nn.Linear(self.flattened_size, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (Tensor): Shape [batch_size, 3, 3, height, width].
        
        Returns:
            Tensor: Predicted steering angle.
        """
        x = self.spatial_temporal_conv(x)
        x = x.view(x.size(0), -1)  # Flatten

        return self.fc(x).squeeze(-1)
