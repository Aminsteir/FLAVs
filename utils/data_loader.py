import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class AutonomousVehicleDataset(Dataset):
    def __init__(self, data_folder, data_file):
        """
        Dataset for autonomous vehicle tasks.

        Args:
            data_folder (str): Path to the folder containing image data.
            data_file (str): Path to the text file mapping images to output angles.
        """
        self.data_folder = data_folder

        # Parse the data file
        self.entries = []
        with open(data_file, "r") as f:
            for line in f.readlines():
                img_name, angle = line.strip().split()[:2]
                self.entries.append((img_name, float(angle)))

    def __len__(self):
        # Ensure dataset can handle triplets
        return len(self.entries) - 2

    def __getitem__(self, idx):
        # Get three consecutive frames for the frame stream
        img_paths = [
            os.path.join(self.data_folder, self.entries[idx + i][0])
            for i in range(3)
        ]
        frames = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in img_paths]

        # Normalize frames to [-1, 1]
        frames = [(frame / 127.5) - 1 for frame in frames]

        # Compute optical flows for the optical flow stream
        optical_flow_1 = self._compute_optical_flow(frames[0], frames[1])  # O_{t-1}
        optical_flow_2 = self._compute_optical_flow(frames[1], frames[2])  # O_t

        # Stack frames for the frame stream input
        frame_stream_input = np.stack(frames, axis=0)  # Shape: (3, H, W, C)

        # Stack optical flows for the optical flow stream input
        optical_flow_stream_input = np.stack([optical_flow_1, optical_flow_2], axis=0)

        # Steering angle (use the angle associated with the latest frame A_t)
        angle = torch.tensor(self.entries[idx + 2][1], dtype=torch.float32)

        # Convert inputs to torch tensors
        frame_stream_input = torch.tensor(frame_stream_input, dtype=torch.float32)
        optical_flow_stream_input = torch.tensor(optical_flow_stream_input, dtype=torch.float32)

        return frame_stream_input, optical_flow_stream_input, angle

    def _compute_optical_flow(self, frame1, frame2):
        """
        Compute optical flow using Gunnar Farneback's algorithm.
        Args:
            frame1 (numpy.ndarray): First frame (RGB).
            frame2 (numpy.ndarray): Second frame (RGB).

        Returns:
            numpy.ndarray: Optical flow image (grayscale with motion information).
        """
        frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)

        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            frame1_gray, frame2_gray, None,
            pyr_scale=0.5, levels=3, winsize=15, iterations=3,
            poly_n=5, poly_sigma=1.2, flags=0
        )

        # Convert optical flow to a representation (magnitude + angle)
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        optical_flow_image = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        return optical_flow_image.astype(np.uint8)
