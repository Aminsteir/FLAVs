import cv2
import numpy as np
import torch

def compute_optical_flow(frame1, frame2):
    """
    Compute optical flow using SimpleFlow algorithm.
    
    Args:
        frame1 (PIL.Image): The first frame.
        frame2 (PIL.Image): The second frame.
        
    Returns:
        torch.Tensor: Normalized optical flow magnitude as a tensor.
    """
    frame1_gray = cv2.cvtColor(np.array(frame1), cv2.COLOR_RGB2GRAY)
    frame2_gray = cv2.cvtColor(np.array(frame2), cv2.COLOR_RGB2GRAY)

    # Compute Farneback optical flow
    flow = cv2.calcOpticalFlowFarneback(
        frame1_gray, frame2_gray, None,
        pyr_scale=0.5, levels=3, winsize=15, iterations=3,
        poly_n=5, poly_sigma=1.1, flags=0
    )

    # Compute magnitude and normalize
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    normalized_flow = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)

    return torch.tensor(normalized_flow, dtype=torch.float32)