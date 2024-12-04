import torch
import torch.nn as nn
import torch.nn.functional as F

class DualStreamModel(nn.Module):
    def __init__(self, input_channels=9, flow_channels=2, image_size=(256, 455), **kwargs):
        """
        Dual-Stream Convolutional Neural Network for autonomous driving.

        Args:
            input_channels (int): Number of channels in the frame stream input (default: 3 frames * 3 channels = 9).
            flow_channels (int): Number of channels in the optical flow stream input (default: 2 flows = 2 channels).
            image_size (tuple): Height and width of the input images.
        """
        super(DualStreamModel, self).__init__()

        # Frame stream (processes RGB frames)
        self.frame_conv1 = nn.Conv2d(input_channels, 12, kernel_size=3, stride=2, padding=1)
        self.frame_conv2 = nn.Conv2d(12, 24, kernel_size=3, stride=2, padding=1)
        self.frame_pool = nn.MaxPool2d(kernel_size=4, stride=2)

        # Optical flow stream (processes optical flow images)
        self.flow_conv1 = nn.Conv2d(flow_channels, 12, kernel_size=3, stride=2, padding=1)
        self.flow_conv2 = nn.Conv2d(12, 24, kernel_size=3, stride=2, padding=1)
        self.flow_pool = nn.MaxPool2d(kernel_size=4, stride=2)

        # Calculate the flattened feature size after convolution and pooling
        self.conv_output_size = self._calculate_conv_output(image_size)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(24 * self.conv_output_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 1)
        )

    def forward(self, frame_input, flow_input):
        """
        Forward pass for the dual-stream model.

        Args:
            frame_input (torch.Tensor): Frame stream input (shape: [batch_size, 9, H, W]).
            flow_input (torch.Tensor): Optical flow stream input (shape: [batch_size, 2, H, W]).

        Returns:
            torch.Tensor: Predicted steering angle.
        """
        # Frame stream processing
        frame = F.elu(self.frame_conv1(frame_input))
        frame = F.elu(self.frame_conv2(frame))
        frame = self.frame_pool(frame)
        frame = torch.flatten(frame, start_dim=1)

        # Flow stream processing
        flow = F.elu(self.flow_conv1(flow_input))
        flow = F.elu(self.flow_conv2(flow))
        flow = self.flow_pool(flow)
        flow = torch.flatten(flow, start_dim=1)

        # Debugging: Print the shapes of each stream
        print(f"Frame stream output shape: {frame.shape}")
        print(f"Flow stream output shape: {flow.shape}")

        # Concatenate the outputs of both streams
        combined = torch.cat((frame, flow), dim=1)

        # Debugging: Print the shape of the combined tensor
        print(f"Combined feature shape: {combined.shape}")

        return self.fc(combined).squeeze(-1)

    def _calculate_conv_output(self, image_size):
        """
        Helper function to calculate the size of the output after convolutions and pooling.

        Args:
            image_size (tuple): (height, width) of the input images.

        Returns:
            int: Flattened size of the convolutional output.
        """
        height, width = image_size
        for _ in range(2):  # Two convolution layers
            height = (height + 2 - 3) // 2 + 1  # Conv2d with kernel=3, stride=2, padding=1
            width = (width + 2 - 3) // 2 + 1
        height = (height - 4) // 2 + 1  # MaxPool2d with kernel=4, stride=2
        width = (width - 4) // 2 + 1
        return height * width