import argparse
import torch
from torch.utils.data import DataLoader
from models.base_model import DualStreamModel
from utils.data_loader import AutonomousVehicleDataset, transform
import torch.nn.functional as F


def train_base_model(data_folder, data_file, save_path, epochs=10, batch_size=32, lr=0.001, device="cpu"):
    """
    Train the base model on the provided dataset.

    Args:
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
    dataset = AutonomousVehicleDataset(data_folder, data_file, transform=transform)

    # Step 2: Split dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Step 3: Initialize the model, optimizer, and loss function
    print("Initializing model...")
    model = DualStreamModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    # Step 4: Training loop
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for frame_input, flow_input, labels in train_loader:
            frame_input, flow_input, labels = (
                frame_input.to(device),
                flow_input.to(device),
                labels.to(device),
            )
            optimizer.zero_grad()
            outputs = model(frame_input, flow_input)
            loss = F.mse_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}")

        # Step 5: Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for frame_input, flow_input, labels in val_loader:
                frame_input, flow_input, labels = frame_input.to(device), flow_input.to(device), labels.to(device)
                outputs = model(frame_input, flow_input)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss:.4f}")

    # Step 6: Save the trained model
    print(f"Saving model to {save_path}...")
    torch.save(model.state_dict(), save_path)
    print("Model saved successfully.")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train Base Model")
    parser.add_argument("--data_folder", type=str, required=True, help="Path to the dataset folder")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the file mapping images to output angles")
    parser.add_argument("--save_path", type=str, default="base_model.pth", help="Path to save the trained model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for optimization")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use ('cpu' or 'cuda')")

    args = parser.parse_args()

    # Train the base model
    train_base_model(
        data_folder=args.data_folder,
        data_file=args.data_file,
        save_path=args.save_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device
    )