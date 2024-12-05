import torch
from torch.utils.data import DataLoader, random_split

from models.registry import get_model

class Worker:
    def __init__(self, model_type, worker_id, dataset, base_model_path=None, split_ratio=0.8, batch_size=16, device='cpu'):
        self.worker_id = worker_id
        self.model_type = model_type
        self.model = get_model(model_type=model_type).to(device)
        self.device = device
        self.batch_size = batch_size
        self.loss_fn = torch.nn.MSELoss()

        # Load pretrained base model if provided
        if base_model_path:
            print(f"Worker {self.worker_id} loading base model from {base_model_path}")
            self.model.load_state_dict(torch.load(base_model_path, map_location=device, weights_only=True))

        # Split the dataset into train/test
        self.train_dataset, self.test_dataset = random_split(dataset, [split_ratio, 1 - split_ratio])

        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

    def train(self, epochs=5, lr=1e-5, subset_ratio=0.25):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.6, 0.99), eps=1e-8, weight_decay=5e-5)
        self.model.train()

        # Sample a subset of the training data -- doesn't need to train on all the data available
        subset, _ = random_split(self.train_dataset, [subset_ratio, 1 - subset_ratio])
        train_loader = DataLoader(subset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                *inputs, targets = batch
                inputs, targets = [inp.to(self.device) for inp in inputs], targets.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(*inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Worker {self.worker_id} - Epoch {epoch + 1}, Loss: {avg_loss:.6f}")
        
        return avg_loss

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.test_loader:
                *inputs, targets = batch
                inputs, targets = [inp.to(self.device) for inp in inputs], targets.to(self.device)
                
                outputs = self.model(*inputs)
                loss = self.loss_fn(outputs, targets)
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

    def save_weights(self, save_path):
        print(f"Saving worker model to {save_path}...")
        torch.save(self.model.state_dict(), save_path)
