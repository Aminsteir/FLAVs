import torch
from models.base_model import DualStreamModel
from workers.worker import Worker
from utils.data_loader import AutonomousVehicleDataset, transform, split_dataset_for_workers
from utils.aggregation import federated_average
import argparse

def centralized_simulation(data_folder, data_file, num_workers=5, rounds=5, epochs_per_worker=3, base_model_path=None, device='cpu'):
    """
    Simulate centralized federated learning.

    Args:
        data_folder (str): Path to the folder containing the dataset.
        data_file (str): Path to the file mapping images to output angles.
        num_workers (int): Number of workers (edge devices).
        rounds (int): Number of federated learning rounds.
        epochs_per_worker (int): Number of local epochs for each worker.
        base_model_path (str): Path to the pretrained base model file.
        device (str): Device to run the simulation ('cpu' or 'cuda').
    """
    # Load the dataset
    dataset = AutonomousVehicleDataset(data_folder, data_file, transform=transform)

    # Split the dataset into equal parts for workers
    worker_datasets = split_dataset_for_workers(dataset, num_workers)

    # Initialize workers with the pretrained base model
    workers = [
        Worker(worker_id=i, model=DualStreamModel(), dataset=worker_datasets[i], base_model_path=base_model_path, device=device)
        for i in range(num_workers)
    ]

    # Initialize the global model
    global_model = DualStreamModel().to(device)
    if base_model_path:
        print(f"Loading global model from {base_model_path}")
        global_model.load_state_dict(torch.load(base_model_path, map_location=device))
    global_weights = global_model.state_dict()

    # Federated learning rounds
    for round_num in range(rounds):
        print(f"\n=== Round {round_num + 1}/{rounds} ===")

        # Local training
        worker_weights = []
        for worker in workers:
            print(f"Worker {worker.worker_id} training...")
            worker.train(epochs=epochs_per_worker)
            worker_weights.append(worker.send_weights())

        # Server aggregates weights
        print("Server aggregating weights...")
        global_weights = federated_average(worker_weights)

        # Distribute updated weights
        global_model.load_state_dict(global_weights)
        for worker in workers:
            worker.update_weights(global_weights)

        # Evaluate global model
        print("Evaluating global model on all workers...")
        for worker in workers:
            worker.evaluate()

    # Save the final global model
    torch.save(global_model.state_dict(), "centralized_global_model.pth")
    print("Global model saved as 'centralized_global_model.pth'.")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Centralized Federated Learning Simulation")
    parser.add_argument("--data_folder", type=str, required=True, help="Path to the dataset folder")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the file mapping images to output angles")
    parser.add_argument("--num_workers", type=int, default=5, help="Number of workers")
    parser.add_argument("--rounds", type=int, default=5, help="Number of federated learning rounds")
    parser.add_argument("--epochs_per_worker", type=int, default=3, help="Number of local epochs per worker")
    parser.add_argument("--base_model_path", type=str, default=None, help="Path to the pretrained base model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use ('cpu' or 'cuda')")

    args = parser.parse_args()

    # Call the simulation function with parsed arguments
    centralized_simulation(
        data_folder=args.data_folder,
        data_file=args.data_file,
        num_workers=args.num_workers,
        rounds=args.rounds,
        epochs_per_worker=args.epochs_per_worker,
        base_model_path=args.base_model_path,
        device=args.device
    )