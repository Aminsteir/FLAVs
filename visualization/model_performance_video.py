import torch
import argparse
import cv2
from models.registry import get_model
import numpy as np
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

    # Draw ground truth wheel
    cv2.circle(extended_image, center_gt, radius, (0, 255, 0), 2)  # Green circle
    angle_gt_rad = np.deg2rad(ground_truth_angle)
    end_point_gt = (
        int(center_gt[0] + radius * np.cos(angle_gt_rad)),
        int(center_gt[1] - radius * np.sin(angle_gt_rad))
    )
    cv2.line(extended_image, center_gt, end_point_gt, (0, 255, 0), 2)
    cv2.putText(extended_image, f"GT: {ground_truth_angle:.2f}", (center_gt[0] - 50, center_gt[1] - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Draw predicted wheel
    cv2.circle(extended_image, center_pred, radius, (0, 0, 255), 2)  # Red circle
    angle_pred_rad = np.deg2rad(predicted_angle)
    end_point_pred = (
        int(center_pred[0] + radius * np.cos(angle_pred_rad)),
        int(center_pred[1] - radius * np.sin(angle_pred_rad))
    )
    cv2.line(extended_image, center_pred, end_point_pred, (0, 0, 255), 2)
    cv2.putText(extended_image, f"Pred: {predicted_angle:.2f}", (center_pred[0] - 50, center_pred[1] - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return extended_image


def create_visualization_video(model, dataset, output_video_path, model_type, device="cpu"):
    """
    Create a video visualizing model predictions vs ground truth.

    Args:
        model (nn.Module): Trained model.
        dataset (Dataset): Dataset for the video.
        output_video_path (str): Path to save the output video.
        model_type (str): Type of model (e.g., "dual_stream", "spatio_temporal").
        device (str): Device to run the model on ("cpu" or "cuda").
    """
    model.eval()
    model.to(device)

    # Video writer setup
    frame_width = 455
    frame_height = 256
    extended_width = frame_width + 200  # Space for overlay
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20,
                                   (extended_width, frame_height))

    for item in dataset:
        *inputs, target = item  # Extract inputs and ground truth
        sin_gt, cos_gt = target[0], target[1]
        frames = inputs[0]  # First input (frame stream)

        # Convert ground truth to angle
        ground_truth_angle = np.rad2deg(np.arctan2(sin_gt, cos_gt))

        # Prepare inputs dynamically based on model type
        if model_type == "dual_stream":
            reshaped_frames = frames.view(3, 3, 256, 455)  # 3 frames, 3 channels
            current_frame = reshaped_frames[-1].cpu().numpy().transpose(1, 2, 0)  # Extract last frame
            current_frame = (current_frame + 1.0) * 127.5  # Rescale [-1, 1] to [0, 255]
            current_frame = np.clip(current_frame, 0, 255).astype(np.uint8)
            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR)
        elif model_type in ["spatio_temporal", "temporal_transformer"]:
            current_frame = frames[-1].cpu().numpy().transpose(1, 2, 0)  # Convert to H x W x C
            current_frame = (current_frame + 1.0) * 127.5
            current_frame = np.clip(current_frame, 0, 255).astype(np.uint8)
            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Add batch dimension and move inputs to the correct device
        inputs = [inp.unsqueeze(0).to(device) for inp in inputs]

        # Perform model inference
        with torch.no_grad():
            outputs = model(*inputs)
            sin_pred, cos_pred = outputs.squeeze(0).cpu().numpy()

        # Convert prediction to angle
        predicted_angle = np.rad2deg(np.arctan2(sin_pred, cos_pred))

        # Draw steering wheels for ground truth and prediction
        visual_frame = draw_steering_wheel(current_frame, ground_truth_angle, predicted_angle)

        # Write frame to video
        video_writer.write(visual_frame)

    video_writer.release()
    print(f"Video saved to {output_video_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a visualization video for model predictions.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file.")
    parser.add_argument("--model_type", type=str, required=True, help="Type of model to load (e.g., dual_stream, spatio_temporal).")
    parser.add_argument("--data_folder", type=str, required=True, help="Path to the test dataset folder.")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the test dataset mapping file.")
    parser.add_argument("--subset_fraction", type=float, default=0.02, help="Fraction of dataset to record video.")
    parser.add_argument("--output_video", type=str, required=True, help="Path to save the output video.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for inference ('cpu' or 'cuda').")

    args = parser.parse_args()

    dataset = AutonomousVehicleDataset(args.data_folder, args.data_file, args.model_type).sample_subset(args.subset_fraction)

    # Load model
    model = get_model(args.model_type)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))

    # Create the video
    create_visualization_video(
        model=model,
        model_type=args.model_type,
        dataset=dataset,
        output_video_path=args.output_video,
        device=args.device
    )
