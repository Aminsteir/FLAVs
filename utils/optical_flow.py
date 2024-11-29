import cv2
import numpy as np
import torch

def compute_optical_flow(frame1, frame2):
    """Compute optical flow using Gunnar Farneback's algorithm."""
    frame1_gray = cv2.cvtColor(np.array(frame1), cv2.COLOR_RGB2GRAY)
    frame2_gray = cv2.cvtColor(np.array(frame2), cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        frame1_gray, frame2_gray, None, 
        pyr_scale=0.5, levels=3, winsize=15, iterations=3, 
        poly_n=5, poly_sigma=1.2, flags=0
    )
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    normalized_flow = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
    return torch.tensor(normalized_flow, dtype=torch.float32)