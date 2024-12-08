from concurrent.futures import ThreadPoolExecutor, as_completed
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
    def __init__(self, data_folder, data_file, model_type, precompute_flow = True, image_size=(256, 455)):
        self.data_folder = data_folder
        self.model_type = model_type
        self.precompute_flow = precompute_flow
        self.image_size = image_size
        self.data = []
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),  # Convert frame to [0, 1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Scale to [-1, 1]
        ])
        
        # Load data file (format: filename steering_angle)
        with open(data_file, "r") as f:
            for line in f.readlines():
                filename, angle = line.strip().split(",")[0].split()[:2]
                angle = float(angle)
                self.data.append((filename, angle))
        
        if model_type == "dual_stream" and precompute_flow:
            self.flow_save_dir = f"{data_folder}/flow/"
            if os.path.exists(self.flow_save_dir):
                print("Folder exists for precomputed flow values. Using existing folder for loader.")
            else:
                # os.makedirs(save_dir, exist_ok=True)
                os.makedirs(self.flow_save_dir)
                self._precompute_optical_flow()

    def _precompute_optical_flow(self):
        """Precompute and save optical flow maps for the entire dataset."""

        def compute_and_save_flow(args):
            """Helper function to compute and save optical flow for a given pair of frames."""
            frame1_path, frame2_path, flow_save_path = args

            # Load frames and compute flow
            frame1 = Image.open(frame1_path).convert("RGB")
            frame2 = Image.open(frame2_path).convert("RGB")
            flow = compute_optical_flow(frame1, frame2)

            # Save optical flow map
            torch.save(flow, flow_save_path)

        print("Precomputing optical flow maps...")

        # Prepare arguments for parallel processing
        tasks = []
        for i in range(len(self.data)):
            frame1_path = os.path.join(self.data_folder, self.data[max(i - 1, 0)][0])
            frame2_path = os.path.join(self.data_folder, self.data[i][0])
            flow_save_path = os.path.join(self.flow_save_dir, f"flow_{i}.pt")
            tasks.append((frame1_path, frame2_path, flow_save_path))

        # Use ThreadPoolExecutor to parallelize the computation
        futures = []
        with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust max_workers based on your CPU cores
            for task in tasks:
                futures.append(executor.submit(compute_and_save_flow, task))
        
            # Explicitly wait for all futures to complete
            for future in as_completed(futures):
                try:
                    future.result()  # Raise any exceptions that occurred during execution
                except Exception as e:
                    print(f"Error occurred: {e}")

        print("Optical flow precomputation completed. Ready to start training.")
            
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
            if self.precompute_flow:
                flow_input = [
                    torch.load(os.path.join(self.flow_save_dir, f"flow_{max(index - 1, 0)}.pt"), weights_only=True),
                    torch.load(os.path.join(self.flow_save_dir, f"flow_{index}.pt"), weights_only=True)
                ]
                flow_input = torch.stack(flow_input, dim=0)  # Shape: [2, H, W]
            else:
                optical_flow_1 = compute_optical_flow(frames[0], frames[1])
                optical_flow_2 = compute_optical_flow(frames[1], frames[2])
                flow_input = torch.stack([optical_flow_1, optical_flow_2], dim=0)  # Shape: [2, H, W]

            frame_input = torch.cat([self.transform(frame) for frame in frames], dim=0)  # Shape: [9, H, W]
            return frame_input, flow_input, target

        elif self.model_type in ["spatio_temporal"]:
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
