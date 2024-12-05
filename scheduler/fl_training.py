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
    # Parameters for the training jobs
    fed_types = ["centralized", "decentralized"]

    data_folder = "data/training_data/data/"
    data_file = "data/training_data/data.txt"
    model_type = "dual_stream"
    base_model_path = "build/dual_stream-base_model.pth"
    save_freq = 10
    num_workers = [4, 8]
    rounds = 50
    epochs_per_worker = 2
    batch_size = 32
    subset_ratio = 0.25
    lr = 1e-5
    device = "cuda"

    for fed_type in fed_types:
        for num_worker in num_workers:
            save_dir = f"build/{fed_type}/{num_worker}_workers/"
            os.makedirs(save_dir, exist_ok=True)

            run_fed_job(
                fed_type, 
                model_type, 
                data_folder, 
                data_file, 
                save_dir, 
                save_freq, 
                num_worker, 
                rounds, 
                epochs_per_worker, 
                batch_size, 
                subset_ratio, 
                lr, 
                base_model_path, 
                device
            )

    print("All fed training jobs completed.")

if __name__ == "__main__":
    main()
