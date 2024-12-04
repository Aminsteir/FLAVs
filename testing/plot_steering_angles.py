import os
import torch
import argparse
import matplotlib.pyplot as plt
from models.registry import get_model
from utils.data_loader import AutonomousVehicleDataset
import numpy as np


def plot_steering_angles(model, dataset, output_plot_path, model_type, device="cpu"):
    model.eval()
    model.to(device)

    ground_truth_angles = []
    predicted_angles = []
    timestamps = []

    for idx, item in enumerate(dataset):
        *inputs, target = item  # Extract inputs and ground truth
        inputs = [inp.unsqueeze(0).to(device) for inp in inputs]
        ground_truth_angle = target.item()

        with torch.no_grad():
            predicted_angle = model(*inputs).squeeze(-1).item()

        timestamps.append(idx)
        ground_truth_angles.append(ground_truth_angle)
        predicted_angles.append(predicted_angle)

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
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot ground truth vs predicted steering angles.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file.")
    parser.add_argument("--data_folder", type=str, required=True, help="Path to the test dataset folder.")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the test dataset mapping file.")
    parser.add_argument("--subset_fraction", type=float, default=0.02, help="Fraction of dataset to use")
    parser.add_argument("--output_plot", type=str, required=True, help="Path to save the output plot.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for inference ('cpu' or 'cuda').")

    args = parser.parse_args()

    dataset = AutonomousVehicleDataset(args.data_folder, args.data_file, args.model_type).sample_subset(args.subset_fraction)

    # Load model
    model = get_model(args.model_type)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))

    # Create the plot
    plot_steering_angles(
        model=model,
        dataset=dataset,
        output_plot_path=args.output_plot,
        model_type=args.model_type,
        device=args.device
    )
