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

    gpu_frame1 = cv2.cuda_GpuMat()
    gpu_frame2 = cv2.cuda_GpuMat()
    gpu_frame1.upload(frame1_gray)
    gpu_frame2.upload(frame2_gray)

    # frame1_gray = cv2.resize(frame1_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    # frame2_gray = cv2.resize(frame2_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    optical_flow = cv2.cuda_FarnebackOpticalFlow.create(
        pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    gpu_flow = optical_flow.calc(gpu_frame1, gpu_frame2, None)
    flow = gpu_flow.download()
    
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    normalized_flow = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
    return torch.tensor(normalized_flow, dtype=torch.float32)