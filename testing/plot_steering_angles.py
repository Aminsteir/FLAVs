import os
import time
import torch
import argparse
import matplotlib.pyplot as plt
from models.registry import get_model
from utils.data_loader import AutonomousVehicleDataset
import numpy as np
from sklearn.metrics import mean_squared_error

def plot_steering_angles(model, dataset, output_plot_path, device="cpu"):
    model.eval()
    model.to(device)

    ground_truth_angles = []
    predicted_angles = []
    timestamps = []

    inference_times = []  # Track inference times
    frame_count = 0

    for idx, item in enumerate(dataset):
        *inputs, target = item  # Extract inputs and ground truth
        inputs = [inp.unsqueeze(0).to(device) for inp in inputs]
        ground_truth_angle = target.item()

        start_time = time.time()
        with torch.no_grad():
            predicted_angle = model(*inputs).item()
        end_time = time.time()

        # Record inference time
        inference_times.append(end_time - start_time)
        frame_count += 1

        timestamps.append(idx)
        ground_truth_angles.append(ground_truth_angle)
        predicted_angles.append(predicted_angle)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(ground_truth_angles, predicted_angles))
    print(f"RMSE: {rmse:.4f}")

    # Calculate and print FPS statistics
    total_time = sum(inference_times)
    avg_time_per_frame = total_time / frame_count
    avg_fps = 1 / avg_time_per_frame
    
    print(f"Inference Stats: Average Time per Frame: {avg_time_per_frame:.4f} seconds")
    print(f"Inference Stats: Average FPS: {avg_fps:.2f}")

    # Plot ground truth vs. predicted angles
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, ground_truth_angles, label="Ground Truth", color="blue", linewidth=1.5)
    plt.plot(timestamps, predicted_angles, label="Predicted", color="red", linestyle="--", linewidth=1.5)
    plt.xlabel("Timestamp")
    plt.ylabel("Steering Wheel Angle (Degrees)")
    plt.title("Comparison of Ground Truth and Predicted Steering Angles")
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig(output_plot_path)
    print(f"Plot saved to {output_plot_path}")
    # plt.show()


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Plot ground truth vs predicted steering angles.")
#     parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file.")
#     parser.add_argument("--model_type", type=str, required=True, help="Type of model to plot.")
#     parser.add_argument("--data_folder", type=str, required=True, help="Path to the test dataset folder.")
#     parser.add_argument("--data_file", type=str, required=True, help="Path to the test dataset mapping file.")
#     parser.add_argument("--subset_fraction", type=float, default=0.02, help="Fraction of dataset to use")
#     parser.add_argument("--output_plot", type=str, required=True, help="Path to save the output plot.")
#     parser.add_argument("--device", type=str, default="cuda", help="Device to use for inference ('cpu' or 'cuda').")

#     args = parser.parse_args()

#     dataset = AutonomousVehicleDataset(args.data_folder, args.data_file, args.model_type).sample_subset(args.subset_fraction)

#     # Load model
#     model = get_model(args.model_type)
#     model.load_state_dict(torch.load(args.model_path, map_location=args.device, weights_only=True))

#     # Create the plot
#     plot_steering_angles(
#         model=model,
#         dataset=dataset,
#         output_plot_path=args.output_plot,
#         device=args.device
#     )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot ground truth vs predicted steering angles.")
    parser.add_argument("--data_folder", type=str, required=True, help="Path to the test dataset folder.")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the test dataset mapping file.")
    parser.add_argument("--output_dir", type=str, default="testing/outputs/", help="Output plot directory")
    parser.add_argument("--subset_fraction", type=float, default=0.02, help="Fraction of dataset to use")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for inference ('cpu' or 'cuda').")

    args = parser.parse_args()

    models = [
        ("dual_stream", "build/dual_stream-base_model.pth"),
        ("spatio_temporal", "build/spatio_temporal-base_model.pth")
    ]

    shared_indices = AutonomousVehicleDataset(args.data_folder, args.data_file, "spatio_temporal").sample_subset(args.subset_fraction).indices
    datasets = []

    for model_info in models:
        model_type, model_path = model_info
        model = get_model(model_type)
        model.load_state_dict(torch.load(model_path, map_location=args.device, weights_only=True))
        model_ds = AutonomousVehicleDataset(args.data_folder, args.data_file, model_type).sample_subset(args.subset_fraction)
        model_ds.indices = shared_indices

        plot_steering_angles(
            model=model,
            dataset=model_ds,
            output_plot_path=os.path.join(args.output_dir, f"{model_type}_{str(args.subset_fraction)}.png"),
            device=args.device
        )