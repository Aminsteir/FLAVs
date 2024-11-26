import torch
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F

class Worker:
    def __init__(self, worker_id, model, dataset, base_model_path=None, split_ratio=0.7, batch_size=16, device='cpu'):
        self.worker_id = worker_id
        self.model = model.to(device)
        self.device = device

        # Load pretrained base model if provided
        if base_model_path:
            print(f"Worker {self.worker_id} loading base model from {base_model_path}")
            self.model.load_state_dict(torch.load(base_model_path, map_location=device))

        # Split the dataset into train/test
        train_size = int(split_ratio * len(dataset))
        test_size = len(dataset) - train_size
        self.train_dataset, self.test_dataset = random_split(dataset, [train_size, test_size])

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

    def train(self, epochs=5, lr=1e-5):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.6, 0.99), eps=1e-8)
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0
            for frame_input, flow_input, labels in self.train_loader:
                frame_input, flow_input, labels = (
                    frame_input.to(self.device),
                    flow_input.to(self.device),
                    labels.to(self.device),
                )
                optimizer.zero_grad()
                outputs = self.model(frame_input, flow_input)
                loss = F.mse_loss(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Worker {self.worker_id} - Epoch {epoch + 1}, Loss: {total_loss / len(self.train_loader):.6f}")

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for frame_input, flow_input, labels in self.test_loader:
                frame_input, flow_input, labels = (
                    frame_input.to(self.device),
                    flow_input.to(self.device),
                    labels.to(self.device),
                )
                outputs = self.model(frame_input, flow_input)
                loss = F.mse_loss(outputs, labels)
                total_loss += loss.item()

        print(f"Worker {self.worker_id} - Test Loss: {total_loss / len(self.test_loader):.6f}")

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
