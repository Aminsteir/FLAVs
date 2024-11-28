import torch
import time
import numpy as np
import argparse
import os
from PIL import Image
import cv2
from torchvision import transforms
from models.registry import get_model


def compute_optical_flow(frame1, frame2):
    """
    Compute optical flow using Gunnar Farneback's algorithm.

    Args:
        frame1 (PIL.Image): First frame.
        frame2 (PIL.Image): Second frame.

    Returns:
        numpy.ndarray: Optical flow magnitude (grayscale).
    """
    frame1_np = np.array(frame1)
    frame2_np = np.array(frame2)

    # Convert frames to grayscale (ensure they are uint8)
    frame1_gray = cv2.cvtColor(frame1_np, cv2.COLOR_RGB2GRAY)
    frame2_gray = cv2.cvtColor(frame2_np, cv2.COLOR_RGB2GRAY)

    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(
        frame1_gray, frame2_gray, None,
        pyr_scale=0.5, levels=3, winsize=15, iterations=3,
        poly_n=5, poly_sigma=1.2, flags=0
    )

    # Compute the magnitude of flow vectors
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Normalize to [0, 1]
    magnitude_normalized = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)

    return magnitude_normalized

def test_inference(model_type, model_path, dataset_path, image_size=(256, 455), num_trials=100, device="cpu"):
    """
    Test the inference time of a trained model.

    Args:
        model_type (str): Type of model to test.
        model_path (str): Path to the trained model file (e.g., "base_model.pth").
        dataset_path (str): Path to the dataset folder containing images.
        image_size (tuple): Size of the input images (height, width).
        num_trials (int): Number of trials to average the inference time.
        device (str): Device to run inference on ("cpu", "cuda", or "mps").
    """
    # Initialize the model
    print(f"Loading {model_type} model...")
    model = get_model(model_type)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Preprocessing for images
    preprocess = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),  # Convert to [0, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])

    # Load images from the dataset folder
    image_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.png'))]
    if len(image_files) < 3:
        raise ValueError("Dataset must contain at least 3 images for frame stream input.")

    print(f"Found {len(image_files)} images in dataset.")
    selected_images = np.random.choice(image_files, size=3 * num_trials, replace=True)

    # Warm-up GPU/CPU
    print("Warming up...")
    for _ in range(10):
        with torch.no_grad():
            if model_type == "dual_stream":
                dummy_frame_input = torch.randn(1, 9, image_size[0], image_size[1], device=device)
                dummy_flow_input = torch.randn(1, 2, image_size[0], image_size[1], device=device)
                model(dummy_frame_input, dummy_flow_input)
            else:
                dummy_input = torch.randn(1, 3, 3, image_size[0], image_size[1], device=device)
                model(dummy_input)

    # Measure inference time
    print("Testing inference time...")
    start_time = time.time()
    for trial in range(num_trials):
        if model_type == "dual_stream":
            # DualStreamModel: Load frames and compute optical flow
            frames = [Image.open(selected_images[trial * 3 + i]) for i in range(3)]
            frame_input = torch.cat(
                [preprocess(frame).unsqueeze(0) for frame in frames], dim=1
            ).to(device)  # Shape: [1, 9, height, width]

            # Compute optical flows
            optical_flow_1 = compute_optical_flow(frames[0], frames[1])  # Shape: [height, width]
            optical_flow_2 = compute_optical_flow(frames[1], frames[2])  # Shape: [height, width]

            # Convert to PyTorch tensors and add the channel dimension
            optical_flow_1 = torch.tensor(optical_flow_1, dtype=torch.float32).unsqueeze(0)  # Shape: [1, height, width]
            optical_flow_2 = torch.tensor(optical_flow_2, dtype=torch.float32).unsqueeze(0)  # Shape: [1, height, width]

            # Stack flows along the channel dimension and add batch dimension
            flow_input = torch.cat([optical_flow_1, optical_flow_2], dim=0).unsqueeze(0).to(device)  # Shape: [1, 2, height, width]

            with torch.no_grad():
                model(frame_input, flow_input)

        elif model_type == "spatio_temporal":
            # SpatioTemporalModel: Stack frames as input
            frames = [preprocess(Image.open(selected_images[trial * 3 + i])) for i in range(3)]
            frame_input = torch.stack(frames).unsqueeze(0).to(device)
            with torch.no_grad():
                model(frame_input)

        elif model_type == "temporal_transformer":
            # TemporalTransformer: Stack frames as input
            frames = [preprocess(Image.open(selected_images[trial * 3 + i])) for i in range(3)]
            frame_input = torch.stack(frames).unsqueeze(0).to(device)
            with torch.no_grad():
                model(frame_input)

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    end_time = time.time()

    # Calculate average inference time
    avg_time = (end_time - start_time) / num_trials
    print(f"Average Inference Time: {avg_time:.6f} seconds per trial ({1 / avg_time:.2f} FPS)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Model Inference")
    parser.add_argument("--model_type", type=str, required=True, help="Type of model to test")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset folder")
    parser.add_argument("--num_trials", type=int, default=100, help="Number of inference trials")
    parser.add_argument("--device", type=str, default="cuda", help="Device (e.g., cuda, cpu, mps)")

    args = parser.parse_args()

    test_inference(
        model_type=args.model_type,
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        num_trials=args.num_trials,
        device=args.device
    )
