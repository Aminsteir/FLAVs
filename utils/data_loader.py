import os
import random
from torch.utils.data import Subset
from utils.optical_flow import compute_optical_flow
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

class AutonomousVehicleDataset(Dataset):
    def __init__(self, data_folder, data_file, model_type, image_size=(256, 455)):
        self.data_folder = data_folder
        self.model_type = model_type
        self.image_size = image_size
        self.data = []
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),  # Convert frame to [0, 1]
        ])
        
        # Load data file (format: filename steering_angle)
        with open(data_file, "r") as f:
            for line in f.readlines():
                filename, angle = line.strip().split(",")[0].split()[:2]
                angle = float(angle)
                self.data.append((filename, angle))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Return a sample based on the model type.

        Args:
            index (int): Index of the sample.

        Returns:
            tuple: Inputs (frames, optional optical flow) and the target steering angle.
        """
        # Load the required frames
        current_item = self.data[index]
        frame_paths = [
            os.path.join(self.data_folder, self.data[max(index - 2, 0)][0]),
            os.path.join(self.data_folder, self.data[max(index - 1, 0)][0]),
            os.path.join(self.data_folder, current_item[0]),
        ]
        frames = [Image.open(fp).convert("RGB") for fp in frame_paths]

        target = torch.tensor(current_item[1], dtype=torch.float32) # load gt_angle in tensor form

        if self.model_type == "dual_stream":
            # Compute optical flow for dual-stream model
            optical_flow_1 = compute_optical_flow(frames[0], frames[1])
            optical_flow_2 = compute_optical_flow(frames[1], frames[2])

            frame_input = torch.cat([self.transform(frame) for frame in frames], dim=0)  # Shape: [9, H, W]
            flow_input = torch.stack([optical_flow_1, optical_flow_2], dim=0)  # Shape: [2, H, W]
            return frame_input, flow_input, target

        elif self.model_type in ["spatio_temporal", "temporal_transformer"]:
            frame_input = torch.stack([self.transform(frame) for frame in frames], dim=0)  # Shape: [3, 3, H, W]
            return frame_input, target

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
    def sample_subset(self, subset_ratio):
        """
        Retrieve a contiguous random subset of the dataset.

        Args:
            subset_ratio (float): Fraction of the dataset to retrieve.

        Returns:
            Dataset: A new Dataset object containing the contiguous subset.
        """
        # Calculate subset size
        subset_size = max(1, int(len(self) * subset_ratio))

        # Randomly select a contiguous block
        start_index = random.randint(0, len(self) - subset_size)

        # Create indices for the subset
        indices = list(range(start_index, start_index + subset_size))

        # Return a new Dataset instance using Subset
        return Subset(self, indices)
