import argparse
import torch
from torch.utils.data import DataLoader
from models.registry import get_model
from utils.data_loader import AutonomousVehicleDataset
import torch.nn.functional as F

from utils.logging_utils import Logger
from utils.loss import circular_loss


def train_base_model(model_type, data_folder, data_file, save_path, split_ratio=0.7, epochs=10, batch_size=32, lr=0.001, device="cpu"):
    """
    Train the base model on the provided dataset.

    Args:
        model (str): Model type for training.
        data_folder (str): Path to the folder containing the dataset.
        data_file (str): Path to the file mapping images to output angles.
        save_path (str): Path to save the trained base model.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        lr (float): Learning rate for optimization.
        device (str): Device to run the training ('cpu' or 'cuda').
    """
    # Step 1: Load the dataset
    print("Loading dataset...")
    dataset = AutonomousVehicleDataset(data_folder, data_file, model_type)

    # Step 2: Split dataset into training and validation sets
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [split_ratio, 1 - split_ratio])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Step 3: Initialize the model, optimizer, and loss function
    print("Initializing model...")
    model = get_model(args.model_type).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.6, 0.99), eps=1e-8)

    # Step 4: Training loop
    print("Starting training...")

    # Initialize logger
    logger = Logger(log_dir="logs", scenario=f"{model_type}_base_model")

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            *inputs, targets = batch
            inputs = [inp.to(device) for inp in inputs]
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(*inputs)
            loss = circular_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}")

        # Log training loss
        logger.log(epoch=epoch + 1, mode="train", loss=avg_train_loss)

        # Step 5: Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                *inputs, targets = batch
                inputs = [inp.to(device) for inp in inputs]
                targets = targets.to(device)

                outputs = model(*inputs)
                loss = circular_loss(outputs, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss:.4f}")

        # Log testing metrics
        logger.log(epoch=epoch + 1, mode="test", loss=avg_val_loss)
    
    # Close logger
    logger.close()

    # Step 6: Save the trained model
    print(f"Saving model to {save_path}...")
    torch.save(model.state_dict(), save_path)
    print("Model saved successfully.")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train Base Model")
    parser.add_argument("--model_type", type=str, default="dual_stream", help="Type of model to use")
    parser.add_argument("--data_folder", type=str, required=True, help="Path to the dataset folder")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the file mapping images to output angles")
    parser.add_argument("--save_path", type=str, default="base_model.pth", help="Path to save the trained model")
    parser.add_argument("--split_ratio", type=float, default=0.7, help="Training/Test Split Ratio")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for optimization")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use ('cpu' or 'cuda')")

    args = parser.parse_args()

    # Train the base model
    train_base_model(
        model_type = args.model_type,
        data_folder=args.data_folder,
        data_file=args.data_file,
        save_path=args.save_path,
        split_ratio=args.split_ratio,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device
    )