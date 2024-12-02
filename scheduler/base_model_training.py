import os
import subprocess
from itertools import product
import time

def run_training_job(model_type, output_type, data_folder, data_file, save_dir, epochs, batch_size, lr, device):
    """
    Function to execute a training job with specified parameters.
    """
    command = [
        "python", "train_base_model.py",
        "--model_type", model_type,
        "--output_type", output_type,
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
    model_types = ["dual_stream", "spatio_temporal", "temporal_transformer"]

    # output_types = ["angle", "angle_norm", "sin_cos"]
    output_types = ["angle"]

    data_folder = "data/base_model_training/data/"
    data_file = "data/base_model_training/data.txt"
    save_dir = "build/"
    epochs = 50
    batch_size = 128
    lr = 1e-4
    device = "cuda"

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Create combinations of model and output types
    job_combinations = list(product(model_types, output_types))

    for model_type, output_type in job_combinations:
        print(f"Started training {model_type} {output_type}")
        print("*"*50)
        
        run_training_job(
            model_type=model_type,
            output_type=output_type,
            data_folder=data_folder,
            data_file=data_file,
            save_dir=save_dir,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device
        )

        print(f"Finished training {model_type} {output_type}")

    print("All training jobs completed.")

if __name__ == "__main__":
    main()
