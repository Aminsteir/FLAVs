import os
import torch
from models.dual_stream import DualStreamModel
from models.registry import get_model
from utils.logging_utils import Logger
from simulations.worker import Worker
from utils.data_loader import AutonomousVehicleDataset
from utils.split_dataset import split_dataset_for_workers
from utils.aggregation import federated_average
from utils.swap_data import swap_worker_data
import argparse

from utils.worker_training_utils import parallelize_workers

def centralized_simulation(model_type, data_folder, data_file, save_dir, save_freq=5, num_workers=10, rounds=4, epochs_per_worker=1, lr=0.00001, subset_ratio=0.2, batch_size=8, base_model_path=None, device='cpu'):
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
    global_model = get_model(model_type).to(device)
    if base_model_path:
        print(f"Loading global model from {base_model_path}")
        global_model.load_state_dict(torch.load(base_model_path, map_location=device, weights_only=True))
    global_weights = global_model.state_dict()

    # Start logger
    logger = Logger(log_dir="logs", scenario=f"centralized_{num_workers}")

    # Federated learning rounds
    for round_num in range(rounds):
        print(f"\n=== Round {round_num + 1}/{rounds} ===")

        # Local training
        worker_weights = []
        train_loss = 0
        for worker in workers:
            print(f"Worker {worker.worker_id} training...")
            train_loss += worker.train(epochs=epochs_per_worker, lr=lr, subset_ratio=subset_ratio)
            worker_weights.append(worker.send_weights())
        
        avg_loss = train_loss / len(workers)

        print(f"Round {round_num + 1} - Average training loss: {avg_loss:.6f}")

        # Server aggregates weights
        print("Server aggregating weights...")
        global_weights = federated_average(worker_weights)

        # U --- Distribute updated weights #
        global_model.load_state_dict(global_weights)

        if (round_num + 1) % save_freq == 0: # Save worker model weights + global
            for worker in workers:
                worker_save_path = os.path.join(save_dir, f"{model_type}-centralized-round_{round_num + 1}-worker_{worker.worker_id}.pth")
                worker.save_weights(worker_save_path)
            if (round_num + 1) < rounds: # Don't save global model if it is the final round -- will get saved anyways
                global_save_path = os.path.join(save_dir, f"{model_type}-centralized-round_{round_num + 1}-global.pth")
                torch.save(global_model.state_dict(), global_save_path)

        for worker in workers:
            worker.update_weights(global_weights)
        # U --- End update weights #

        # Log round-level metrics
        logger.log(epoch=round_num + 1, mode="train", loss=avg_loss)

        # Log testing metrics after aggregation
        test_loss = 0
        for worker in workers:
            test_loss += worker.evaluate()
        avg_loss = test_loss / len(workers)

        print(f"Round {round_num + 1} - Average test loss: {avg_loss:.6f}")

        logger.log(epoch=round_num + 1, mode="test", loss=avg_loss)

    logger.close()

    # Save the final global model
    global_save_path = os.path.join(save_dir, f"{model_type}-centralized-finished-global.pth")
    torch.save(global_model.state_dict(), global_save_path)
    print(f"Final global model saved at: '{global_save_path}'.")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Centralized Federated Learning Simulation")
    parser.add_argument("--model_type", type=str, default="dual_stream", help="Model type to train")
    parser.add_argument("--data_folder", type=str, required=True, help="Path to the dataset folder")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the file mapping images to output angles")
    parser.add_argument("--save_dir", type=str, default="build/", help="Output directory for saving trained models")
    parser.add_argument("--save_freq", type=int, default=5, help="Frequency to save worker models (e.g., 5 --> every 5 rounds)")
    parser.add_argument("--num_workers", type=int, default=5, help="Number of workers")
    parser.add_argument("--rounds", type=int, default=5, help="Number of federated learning rounds")
    parser.add_argument("--epochs_per_worker", type=int, default=3, help="Number of local epochs per worker")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training and validation")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for training")
    parser.add_argument("--subset_ratio", type=float, default=0.1, help="Fraction of training data to learn from each round")
    parser.add_argument("--base_model_path", type=str, default=None, help="Path to the pretrained base model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use ('cpu' or 'cuda')")

    args = parser.parse_args()

    # Call the simulation function with parsed arguments
    centralized_simulation(
        model_type=args.model_type,
        data_folder=args.data_folder,
        data_file=args.data_file,
        save_dir=args.save_dir,
        save_freq=args.save_freq,
        num_workers=args.num_workers,
        rounds=args.rounds,
        epochs_per_worker=args.epochs_per_worker,
        batch_size=args.batch_size,
        subset_ratio=args.subset_ratio,
        lr=args.lr,
        base_model_path=args.base_model_path,
        device=args.device
    )