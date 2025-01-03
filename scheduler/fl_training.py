import argparse
import os
import subprocess
from itertools import product
import time

def run_fed_job(fed_type, model_type, data_folder, data_file, save_dir, save_freq, num_workers, rounds, epochs_per_worker, batch_size, subset_ratio, lr, base_model_path, device):
    """
    Function to execute a training job with specified parameters.
    """
    if fed_type == "centralized":
        command = [
            "python", "simulations/centralized.py",
            "--model_type", model_type,
            "--data_folder", data_folder,
            "--data_file", data_file,
            "--save_dir", save_dir,
            "--save_freq", str(save_freq),
            "--num_workers", str(num_workers),
            "--rounds", str(rounds),
            "--epochs_per_worker", str(epochs_per_worker),
            "--batch_size", str(batch_size),
            "--lr", str(lr),
            "--subset_ratio", str(subset_ratio),
            "--base_model_path", base_model_path,
            "--device", device
        ]
    elif fed_type == "decentralized":
        command = [
            "python", "simulations/decentralized.py",
            "--model_type", model_type,
            "--data_folder", data_folder,
            "--data_file", data_file,
            "--save_dir", save_dir,
            "--save_freq", str(save_freq),
            "--num_workers", str(num_workers),
            "--rounds", str(rounds),
            "--epochs_per_worker", str(epochs_per_worker),
            "--batch_size", str(batch_size),
            "--lr", str(lr),
            "--subset_ratio", str(subset_ratio),
            "--base_model_path", base_model_path,
            "--device", device
        ]
    else:
        raise ValueError(f"Unsupported federated learning type: {fed_type}")
    
    print(f"Running: {' '.join(command)}")
    subprocess.run(command)


def main():
    parser = argparse.ArgumentParser(description="Federated Learning Scheduler")
    parser.add_argument("--fed_type", type=str, required=True, help="Federated learning training type")
    parser.add_argument("--num_workers", type=int, default=5, help="Number of workers used in training")
    parser.add_argument("--subset_ratio", type=float, default=0.15, help="Subset of worker dataset that each worker trains from each round.")

    args = parser.parse_args()

    data_folder = "data/training_data/data/"
    data_file = "data/training_data/data.txt"
    model_type = "dual_stream"
    base_model_path = "build/dual_stream-base_model.pth"

    save_freq = 10
    rounds = 50
    epochs_per_worker = 4
    batch_size = 32
    subset_ratio = args.subset_ratio
    lr = 2e-5
    device = "cuda"

    save_dir = f"build/{args.fed_type}/{args.num_workers}_workers/"
    os.makedirs(save_dir, exist_ok=True)

    run_fed_job(
        args.fed_type, 
        model_type, 
        data_folder, 
        data_file, 
        save_dir, 
        save_freq, 
        args.num_workers, 
        rounds, 
        epochs_per_worker, 
        batch_size, 
        subset_ratio, 
        lr, 
        base_model_path, 
        device
    )

    print("Training job completed.")

if __name__ == "__main__":
    main()
