import os
import subprocess
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

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

def run_training_job(model_type, output_type, data_folder, data_file, save_dir, epochs, batch_size, lr, device, memory_requirement):
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
    print(f"Running: {' '.join(command)} (requires {memory_requirement} MiB)")
    subprocess.run(command)

def main():
    # Parameters for the training jobs
    model_types = ["dual_stream", "spatio_temporal", "temporal_transformer"]
    output_types = ["angle", "angle_norm", "sin_cos"]
    memory_requirements = {
        "dual_stream": 3700,
        "spatio_temporal": 6200,
        "temporal_transformer": 3700,
    }
    data_folder = "data/base_model_training/data/"
    data_file = "data/base_model_training/data.txt"
    save_dir = "build/"
    epochs = 25
    batch_size = 128
    lr = 0.0001
    device = "cuda"

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Create combinations of model and output types
    job_combinations = list(product(model_types, output_types))

    # Track running jobs and available memory
    active_jobs = []
    total_memory = check_available_gpu_memory()
    print(f"Total GPU memory: {total_memory} MiB")

    # Function to check if enough memory is available for a job
    def can_schedule_job(required_memory):
        available_memory = check_available_gpu_memory()
        used_memory = sum(job["memory"] for job in active_jobs)
        return available_memory - used_memory >= required_memory

    # Thread pool for job scheduling
    with ThreadPoolExecutor(max_workers=len(job_combinations)) as executor:
        futures = []
        for model_type, output_type in job_combinations:
            memory_requirement = memory_requirements[model_type]

            # Wait until there's enough memory to schedule the job
            while not can_schedule_job(memory_requirement):
                print("Waiting for available GPU memory...")
                time.sleep(5)  # Wait 5 seconds before checking again

            # Schedule the job
            future = executor.submit(
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
                memory_requirement=memory_requirement
            )
            futures.append(future)
            active_jobs.append({"future": future, "memory": memory_requirement})

            # Remove completed jobs from active_jobs
            for job in active_jobs[:]:
                if job["future"].done():
                    active_jobs.remove(job)

        # Wait for all jobs to finish
        for future in as_completed(futures):
            future.result()

    print("All training jobs completed.")

if __name__ == "__main__":
    main()
