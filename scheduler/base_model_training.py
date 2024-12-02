import os
import subprocess
from itertools import product
import torch

def run_training_job(model_type, output_type, data_folder, data_file, save_dir, epochs, batch_size, lr, device):
    """
    Function to execute a training job with specified parameters.
    
    Args:
        model_type (str): The architecture of the model to use.
        output_type (str): The format of the model output.
        data_folder (str): Path to the dataset folder.
        data_file (str): Path to the dataset mapping file.
        save_dir (str): Directory to save the trained model.
        epochs (int): Number of epochs for training.
        batch_size (int): Batch size for training.
        lr (float): Learning rate for training.
        device (str): Device to use ('cpu' or 'cuda').
    """
    save_path = os.path.join(save_dir, f"{model_type}-{output_type}-base_model.pth")
    
    # Construct the command to execute the training script
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
    
    # Print the command for logging/debugging
    print(f"Running: {' '.join(command)}")
    
    # Execute the command
    subprocess.run(command)

def main():
    # Define the parameters for the scheduler
    model_types = ["dual_stream", "spatio_temporal", "temporal_transformer"]  # Model architectures
    output_types = ["angle", "angle_norm", "sin_cos"]  # Output types
    data_folder = "data/base_model_training/data/"
    data_file = "data/base_model_training/data.txt"
    save_dir = "build/"
    epochs = 25
    batch_size = 32
    lr = 0.0001
    device = "cuda"

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Iterate over all combinations of model types and output types
    for model_type, output_type in product(model_types, output_types):
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

if __name__ == "__main__":
    main()
