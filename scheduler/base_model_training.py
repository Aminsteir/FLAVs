import os
import subprocess
from itertools import product
from concurrent.futures import ThreadPoolExecutor
import time

# Estimate GPU memory usage per job
JOB_MEMORY_USAGE_MIB = 3700  # Approximate memory per job in MiB
GPU_TOTAL_MEMORY_MIB = 20470  # Total GPU memory in MiB

def check_available_gpu_memory():
    """
    Check available GPU memory using nvidia-smi.
    
    Returns:
        int: Available GPU memory in MiB.
    """
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"],
            encoding="utf-8"
        )
        return int(output.strip().split("\n")[0])
    except Exception as e:
        print(f"Error checking GPU memory: {e}")
        return 0

def run_training_job(model_type, output_type, data_folder, data_file, save_dir, epochs, batch_size, lr, device):
    """
    Function to execute a training job with specified parameters.
    """
    # save_path = os.path.join(save_dir, f"{model_type}-{output_type}-base_model.pth")
    
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
    output_types = ["angle", "angle_norm", "sin_cos"]
    data_folder = "data/base_model_training/data/"
    data_file = "data/base_model_training/data.txt"
    save_dir = "build/"
    epochs = 25
    batch_size = 128
    lr = 0.0001
    device = "cuda"

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Calculate max parallel jobs based on memory
    available_memory = check_available_gpu_memory()
    max_parallel_jobs = max(1, available_memory // JOB_MEMORY_USAGE_MIB)
    print(f"Available GPU memory: {available_memory} MiB")
    print(f"Max parallel jobs: {max_parallel_jobs}")

    # Create combinations of model and output types
    job_combinations = list(product(model_types, output_types))

    # Schedule jobs in parallel
    with ThreadPoolExecutor(max_workers=max_parallel_jobs) as executor:
        futures = []
        for model_type, output_type in job_combinations:
            futures.append(
                executor.submit(
                    run_training_job,
                    model_type=model_type,
                    output_type=output_type,
                    data_folder=data_folder,
                    data_file=data_file,
                    save_dir=save_dir,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    device=device,
                )
            )
        
        # Wait for all jobs to finish
        for future in futures:
            future.result()

    print("All training jobs completed.")

if __name__ == "__main__":
    main()