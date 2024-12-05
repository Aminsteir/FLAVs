import os
import time
import torch
import argparse
import cv2
import numpy as np
from models.registry import get_model
from utils.data_loader import AutonomousVehicleDataset

def draw_steering_wheel(image, ground_truth_angle, predicted_angle):
    """
    Draws steering wheels for ground truth and predicted angles on the image.

    Args:
        image (numpy.ndarray): Original image (H x W x 3).
        ground_truth_angle (float): Ground truth steering angle in degrees.
        predicted_angle (float): Predicted steering angle in degrees.

    Returns:
        numpy.ndarray: Image with steering wheels drawn.
    """
    height, width, _ = image.shape

    # Create space for the graphics (extend the frame horizontally)
    extended_image = np.zeros((height, width + 200, 3), dtype=np.uint8)
    extended_image[:, :width] = image  # Place the original image on the left

    # Define wheel properties
    center_gt = (width + 100, height // 3)  # Position of ground truth wheel
    center_pred = (width + 100, 2 * height // 3)  # Position of predicted wheel
    radius = 50

    def angle_to_coordinates(center, radius, angle_deg):
        """
        Convert an angle in degrees to x, y coordinates for the line endpoint.
        Adjust to make 0° point upwards (12 o'clock position).
        """
        angle_rad = np.deg2rad(-angle_deg)  # Flip the angle to correct direction
        x = int(center[0] - radius * np.sin(angle_rad))  # Multiply x by -1 for correct direction
        y = int(center[1] - radius * np.cos(angle_rad))  # Subtract to flip the y-axis
        return x, y

    # Draw ground truth wheel
    cv2.circle(extended_image, center_gt, radius, (0, 255, 0), 2)  # Green circle
    end_point_gt = angle_to_coordinates(center_gt, radius, ground_truth_angle)
    cv2.line(extended_image, center_gt, end_point_gt, (0, 255, 0), 2)
    cv2.putText(extended_image, f"GT: {ground_truth_angle:.2f}", (center_gt[0] - 50, center_gt[1] - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Draw predicted wheel
    cv2.circle(extended_image, center_pred, radius, (0, 0, 255), 2)  # Red circle
    end_point_pred = angle_to_coordinates(center_pred, radius, predicted_angle)
    cv2.line(extended_image, center_pred, end_point_pred, (0, 0, 255), 2)
    cv2.putText(extended_image, f"Pred: {predicted_angle:.2f}", (center_pred[0] - 50, center_pred[1] - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return extended_image

def create_visualization_video(model, dataset, output_video_path, model_type, fps=30.0, device="cpu"):
    model.eval()
    model.to(device)

    # Video writer setup
    frame_width = 455
    frame_height = 256
    extended_width = frame_width + 200  # Space for overlay
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                   (extended_width, frame_height))
    
    inference_times = []  # Track inference times
    frame_count = 0

    for item in dataset:
        *inputs, target = item  # Extract inputs and ground truth
        frames = inputs[0]  # First input (frame stream)

        ground_truth_angle = target.item()

        if model_type == "dual_stream":
            frames = frames.view(3, 3, 256, 455)  # 3 frames, 3 channels

        current_frame = frames[-1].cpu().numpy().transpose(1, 2, 0)  # Convert to H x W x C
        current_frame = (current_frame + 1) * 127.5
        current_frame = np.clip(current_frame, 0, 255).astype(np.uint8)
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR)

        # Add batch dimension and move inputs to the correct device
        inputs = [inp.unsqueeze(0).to(device) for inp in inputs]

        # Perform model inference and measure time
        start_time = time.time()
        with torch.no_grad():
            predicted_angle = model(*inputs).item()
        end_time = time.time()

        # Record inference time
        inference_times.append(end_time - start_time)
        frame_count += 1

        # Draw steering wheels for ground truth and prediction
        visual_frame = draw_steering_wheel(current_frame, ground_truth_angle, predicted_angle)

        # Write frame to video
        video_writer.write(visual_frame)

    video_writer.release()

    # Calculate and print FPS statistics
    total_time = sum(inference_times)
    avg_time_per_frame = total_time / frame_count
    avg_fps = 1 / avg_time_per_frame

    print(f"Video saved to {output_video_path}")
    print(f"Inference Stats: Average Time per Frame: {avg_time_per_frame:.4f} seconds")
    print(f"Inference Stats: Average FPS: {avg_fps:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a visualization video for model predictions.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file.")
    parser.add_argument("--model_type", type=str, required=True, help="Model architecture to load (e.g., temporal_transformer).")
    parser.add_argument("--data_folder", type=str, required=True, help="Path to the test dataset folder.")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the test dataset mapping file.")
    parser.add_argument("--subset_fraction", type=float, default=0.02, help="Fraction of dataset to record video.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output video.")
    parser.add_argument("--fps", type=float, default=30.0, help="FPS (frames per second) of output video")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for inference ('cpu' or 'cuda').")

    args = parser.parse_args()

    output_video_path = os.path.join(args.output_dir, f"{args.model_type}-driving_visualization.mp4")

    dataset = AutonomousVehicleDataset(args.data_folder, args.data_file, args.model_type, precompute_flow=False).sample_subset(args.subset_fraction)

    # Load model
    model = get_model(model_type=args.model_type)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device, weights_only=True))

    # Create the video
    create_visualization_video(
        model=model,
        model_type=args.model_type,
        dataset=dataset,
        output_video_path=output_video_path,
        fps=args.fps,
        device=args.device
    )