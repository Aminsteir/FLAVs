import os
import torch
import argparse
import random
from utils.logging_utils import Logger
from utils.swap_data import swap_worker_data
from simulations.worker import Worker
from utils.data_loader import AutonomousVehicleDataset
from utils.split_dataset import split_dataset_for_workers
from utils.aggregation import federated_average

def decentralized_simulation(model_type, data_folder, data_file, save_dir, save_freq=5, num_workers=5, rounds=5, epochs_per_worker=3, lr=1e-5, subset_ratio=0.2, batch_size=8, base_model_path=None, device='cpu'):
    # Step 1: Load the dataset
    dataset = AutonomousVehicleDataset(data_folder, data_file, model_type)

    # Step 2: Split the dataset into equal parts for workers
    worker_datasets = split_dataset_for_workers(dataset, num_workers)

    # Step 3: Initialize workers with the pretrained base model
    workers = [
        Worker(worker_id=i, model_type=model_type, dataset=worker_datasets[i], base_model_path=base_model_path, batch_size=batch_size, device=device)
        for i in range(num_workers)
    ]

    # Start logger
    logger = Logger(log_dir="logs", scenario=f"decentralized_{num_workers}")

    # Step 4: Decentralized learning rounds
    for round_num in range(rounds):
        print(f"\n=== Round {round_num + 1}/{rounds} ===")

        # Step 4.1: Generate dynamic neighbor map
        neighbors_map = {
            i: random.sample(
                [j for j in range(num_workers) if j != i],  # Exclude self
                min(num_workers - 1, random.randint(1, 4))  # Randomly select 2 to 4 peers
            )
            for i in range(num_workers)
        }
        print(f"Dynamic Neighbor Map for Round {round_num + 1}: {neighbors_map}")

        # Step 4.2: Local training
        train_loss = 0
        for worker in workers:
            print(f"Worker {worker.worker_id} training...")
            train_loss += worker.train(epochs=epochs_per_worker, lr=lr, subset_ratio=subset_ratio)

        avg_loss = train_loss / len(workers)

        print(f"Round {round_num + 1} - Avg. train loss: {avg_loss:.6f}")

        print("Aggegrating model weights with neighbors...")

        # Step 4.3: Peer-to-peer weight sharing and averaging
        updated_weights = {}
        for worker in workers:
            peer_weights = [workers[peer_id].send_weights() for peer_id in neighbors_map[worker.worker_id]]
            local_weights = worker.send_weights()
            new_weights = federated_average([local_weights] + peer_weights)
            updated_weights[worker.worker_id] = new_weights

        # Step 4.4: Update weights for all workers
        for worker in workers:
            worker.update_weights(updated_weights[worker.worker_id])

        if (round_num + 1) % save_freq == 0 and (round_num + 1) < rounds: # Save worker model weights, don't if it's the last round
            for worker in workers:
                worker_save_path = os.path.join(save_dir, f"{model_type}-decentralized-round_{round_num + 1}-worker_{worker.worker_id}.pth")
                worker.save_weights(worker_save_path)
        
        # Log round-level metrics
        logger.log(epoch=round_num + 1, mode="train", loss=avg_loss)

        # Log testing metrics after aggregation
        test_loss = 0
        for worker in workers:
            test_loss += worker.evaluate()
        avg_loss = test_loss / len(workers)

        print(f"Round {round_num + 1} - Avg. test loss: {avg_loss:.6f}")

        logger.log(epoch=round_num + 1, mode="test", loss=avg_loss)

    logger.close()

    # Step 5: Save final models
    for worker in workers:
        worker_save_path = os.path.join(save_dir, f"{model_type}-decentralized-finished-worker_{worker.worker_id}.pth")
        worker.save_weights(worker_save_path)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Decentralized Federated Learning Simulation")
    parser.add_argument("--model_type", type=str, default="dual_stream", help="Type of model to train")
    parser.add_argument("--data_folder", type=str, required=True, help="Path to the dataset folder")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the file mapping images to output angles")
    parser.add_argument("--save_dir", type=str, default="build/", help="Output directory for saving trained models")
    parser.add_argument("--save_freq", type=int, default=5, help="Frequency to save worker models (e.g., 5 --> every 5 rounds)")
    parser.add_argument("--num_workers", type=int, default=5, help="Number of workers")
    parser.add_argument("--rounds", type=int, default=5, help="Number of federated learning rounds")
    parser.add_argument("--epochs_per_worker", type=int, default=3, help="Number of local epochs per worker")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training and validation")
    parser.add_argument("--subset_ratio", type=float, default=0.1, help="Subset of training data to train each round")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for training")
    parser.add_argument("--base_model_path", type=str, default=None, help="Path to the pretrained base model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use ('cpu' or 'cuda')")

    args = parser.parse_args()

    # Call the simulation function with parsed arguments
    decentralized_simulation(
        model_type=args.model_type,
        data_folder=args.data_folder,
        data_file=args.data_file,
        save_dir=args.save_dir,
        save_freq=args.save_freq,
        num_workers=args.num_workers,
        rounds=args.rounds,
        subset_ratio=args.subset_ratio,
        lr=args.lr,
        epochs_per_worker=args.epochs_per_worker,
        base_model_path=args.base_model_path,
        device=args.device
    )
