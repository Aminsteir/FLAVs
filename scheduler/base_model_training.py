import os
import subprocess
from itertools import product
import time

def run_training_job(model_type, data_folder, data_file, save_dir, epochs, batch_size, lr, device):
    """
    Function to execute a training job with specified parameters.
    """
    command = [
        "python", "train_base_model.py",
        "--model_type", model_type,
        "--data_folder", data_folder,
        "--data_file", data_file,
        "--save_dir", save_dir,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--lr", str(lr),
        "--device", device
    ]
    print(f"Running: {' '.join(command)}")
    subprocess.run(command)

def main():
    # Parameters for the training jobs
    models = {"dual_stream": 1024, "spatio_temporal": 512, "temporal_transformer": 128}

    data_folder = "data/base_model_training/data/"
    data_file = "data/base_model_training/data.txt"
    save_dir = "build/"
    epochs = 25
    lr = 4e-4
    device = "cuda"

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    for model in models:
        batch_size = models[model]
        run_training_job(
            model_type=model,
            data_folder=data_folder,
            data_file=data_file,
            save_dir=save_dir,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device
        )

    print("All training jobs completed.")

if __name__ == "__main__":
    main()
