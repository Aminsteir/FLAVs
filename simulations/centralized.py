import torch
from models.dual_stream import DualStreamModel
from utils.logging_utils import Logger
from workers.worker import Worker
from utils.data_loader import AutonomousVehicleDataset
from utils.split_dataset import split_dataset_for_workers
from utils.aggregation import federated_average
from utils.swap_data import swap_worker_data
import argparse

def centralized_simulation(model_type, data_folder, data_file, num_workers=10, rounds=4, epochs_per_worker=1, lr=0.00001, subset_ratio=0.2, batch_size=8, base_model_path=None, device='cpu'):
    """
    Simulate centralized federated learning.

    Args:
        model_type (str): Model type for training.
        data_folder (str): Path to the folder containing the dataset.
        data_file (str): Path to the file mapping images to output angles.
        num_workers (int): Number of workers (edge devices).
        rounds (int): Number of federated learning rounds.
        epochs_per_worker (int): Number of local epochs for each worker.
        lr (float): The learning rate of the workers in the simulation.
        subset_ratio (float): Each round, the fraction of the dataset the worker should train on.
        base_model_path (str): Path to the pretrained base model file.
        device (str): Device to run the simulation ('cpu' or 'cuda').
    """
    # Load the dataset
    dataset = AutonomousVehicleDataset(data_folder, data_file, model_type)

    # Split the dataset into equal parts for workers
    worker_datasets = split_dataset_for_workers(dataset, num_workers)

    # Initialize workers with the pretrained base model
    workers = [
        Worker(worker_id=i, model_type=model_type, dataset=worker_datasets[i], base_model_path=base_model_path, batch_size=batch_size, device=device)
        for i in range(num_workers)
    ]

    # Initialize the global model
    global_model = DualStreamModel().to(device)
    if base_model_path:
        print(f"Loading global model from {base_model_path}")
        global_model.load_state_dict(torch.load(base_model_path, map_location=device))
    global_weights = global_model.state_dict()

    # Start logger
    logger = Logger(log_dir="logs", scenario="centralized")

    # Federated learning rounds
    for round_num in range(rounds):
        print(f"\n=== Round {round_num + 1}/{rounds} ===")

        # Local training
        worker_weights = []
        test_loss = 0
        for worker in workers:
            print(f"Worker {worker.worker_id} training...")
            test_loss += worker.train(epochs=epochs_per_worker, lr=lr, subset_ratio=subset_ratio)
            worker_weights.append(worker.send_weights())
        
        avg_loss = test_loss / len(workers)

        # Server aggregates weights
        print("Server aggregating weights...")
        global_weights = federated_average(worker_weights)

        # Distribute updated weights
        global_model.load_state_dict(global_weights)
        for worker in workers:
            worker.update_weights(global_weights)

        # Randomly swap worker data -- simulate new environments
        swap_worker_data(workers)

        # Log round-level metrics
        logger.log(epoch=round + 1, mode="train", loss=avg_loss)

        # Log testing metrics after aggregation
        test_loss = 0
        for worker in workers:
            test_loss += worker.evaluate()
        avg_loss = test_loss / len(workers)

        logger.log(epoch=round + 1, mode="test", loss=avg_loss)

    logger.close()

    # Save the final global model
    torch.save(global_model.state_dict(), "centralized_global_model.pth")
    print("Global model saved as 'centralized_global_model.pth'.")


def evaluate_global_model(global_model, val_loader, device):
    criterion = torch.nn.MSELoss()
    global_model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            *inputs, labels = batch
            inputs = [inp.to(device) for inp in inputs]
            labels = labels.to(device)

            outputs = global_model(*inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

    avg_val_loss = test_loss / len(val_loader)
    return avg_val_loss  


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Centralized Federated Learning Simulation")
    parser.add_argument("--model_type", type=str, default="dual_stream", help="Model type to train")
    parser.add_argument("--data_folder", type=str, required=True, help="Path to the dataset folder")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the file mapping images to output angles")
    parser.add_argument("--num_workers", type=int, default=5, help="Number of workers")
    parser.add_argument("--rounds", type=int, default=5, help="Number of federated learning rounds")
    parser.add_argument("--epochs_per_worker", type=int, default=3, help="Number of local epochs per worker")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training and validation")
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
        batch_size=args.batch_size,
        base_model_path=args.base_model_path,
        device=args.device
    )