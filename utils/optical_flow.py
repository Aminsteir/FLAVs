import cv2
import numpy as np
import torch

# def compute_optical_flow(frame1, frame2, scale=0.5):
#     """
#     Compute optical flow using Gunnar Farneback's algorithm with optimizations.

#     Args:
#         frame1 (PIL.Image): First frame.
#         frame2 (PIL.Image): Second frame.
#         scale (float): Scale factor to downsample the frames for faster computation.

#     Returns:
#         torch.Tensor: Normalized optical flow magnitude.
#     """
#     # Convert frames to grayscale and resize for faster computation
#     frame1_gray = cv2.cvtColor(np.array(frame1), cv2.COLOR_RGB2GRAY)
#     frame2_gray = cv2.cvtColor(np.array(frame2), cv2.COLOR_RGB2GRAY)
#     frame1_gray = cv2.resize(frame1_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
#     frame2_gray = cv2.resize(frame2_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

#     # Compute optical flow
#     flow = cv2.calcOpticalFlowFarneback(
#         frame1_gray, frame2_gray, None, 
#         pyr_scale=0.5, levels=2, winsize=9, iterations=2, 
#         poly_n=5, poly_sigma=1.1, flags=0
#     )

#     # Normalize flow magnitude to [0, 1]
#     magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
#     normalized_flow = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)

#     # Upscale to original size and return as a tensor
#     original_size = (frame1.size[0], frame1.size[1])
#     normalized_flow = cv2.resize(normalized_flow, original_size, interpolation=cv2.INTER_LINEAR)
#     return torch.tensor(normalized_flow, dtype=torch.float32)

def compute_optical_flow(frame1, frame2, scale = 0.5):
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

    frame1_gray = cv2.resize(frame1_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    frame2_gray = cv2.resize(frame2_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
    # Compute optical flow using SimpleFlow
    flow = cv2.optflow.calcOpticalFlowSF(
        frame1_gray, frame2_gray, layers=3, averaging_block_size=2, max_flow=4
    )
    
    # Compute magnitude and normalize
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    normalized_flow = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)

    # Upscale to original size and return as a tensor
    # original_size = (frame1.size[0], frame1.size[1])
    # normalized_flow = cv2.resize(normalized_flow, original_size, interpolation=cv2.INTER_LINEAR)
    
    return torch.tensor(normalized_flow, dtype=torch.float32)