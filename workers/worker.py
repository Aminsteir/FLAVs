import torch
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from models.registry import get_model
from utils.split_dataset import sample_subset

class Worker:
    def __init__(self, model_type, worker_id, dataset, base_model_path=None, split_ratio=0.8, batch_size=16, device='cpu'):
        self.worker_id = worker_id
        self.model = get_model(model_type).to(device)
        self.device = device
        self.batch_size = batch_size

        # Load pretrained base model if provided
        if base_model_path:
            print(f"Worker {self.worker_id} loading base model from {base_model_path}")
            self.model.load_state_dict(torch.load(base_model_path, map_location=device))

        # Split the dataset into train/test
        train_size = int(split_ratio * len(dataset))
        test_size = len(dataset) - train_size
        self.train_dataset, self.test_dataset = random_split(dataset, [train_size, test_size])

        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

    def train(self, epochs=5, lr=1e-5, subset_ratio=0.2):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.6, 0.99), eps=1e-8)
        self.model.train()

        # Sample a subset of the training data -- doesn't need to train on all the data available
        subset = sample_subset(self.train_dataset, subset_ratio)
        train_loader = DataLoader(subset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                *inputs, labels = batch
                inputs = [inp.to(self.device) for inp in inputs]
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(*inputs)
                loss = F.mse_loss(outputs.squeeze(-1), labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            print(f"Worker {self.worker_id} - Epoch {epoch + 1}, Loss: {avg_loss:.6f}")
        
        return avg_loss

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.test_loader:
                *inputs, labels = batch
                inputs = [inp.to(self.device) for inp in inputs]
                labels = labels.to(self.device)
                
                outputs = self.model(*inputs)
                loss = F.mse_loss(outputs.squeeze(-1), labels)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.test_loader)

        print(f"Worker {self.worker_id} - Test Loss: {avg_loss:.6f}")
        return avg_loss

    def send_weights(self):
        """
        Send the worker's model weights.

        Returns:
            dict: Model state dictionary.
        """
        return {k: v.clone().detach() for k, v in self.model.state_dict().items()}

    def update_weights(self, new_weights):
        """
        Update the worker's model with new weights.

        Args:
            new_weights (dict): New model state dictionary.
        """
        self.model.load_state_dict(new_weights)
