import os
import logging
import csv
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_dir="logs", scenario="base_model"):
        """
        General-purpose logger for training and evaluation.

        Args:
            log_dir (str): Base directory for logs.
            scenario (str): Name of the scenario (e.g., "base_model", "decentralized", "centralized").
        """
        self.log_dir = os.path.join(log_dir, scenario)
        os.makedirs(self.log_dir, exist_ok=True)

        # File logger
        self.file_logger = logging.getLogger(scenario)
        self.file_logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(os.path.join(self.log_dir, f"{scenario}.log"))
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        self.file_logger.addHandler(file_handler)

        # CSV logger
        self.csv_file = os.path.join(self.log_dir, f"{scenario}.csv")
        with open(self.csv_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Mode", "Loss", "Accuracy"])  # CSV header

        # TensorBoard writer
        self.tensorboard_writer = SummaryWriter(log_dir=self.log_dir)

    def log_to_file(self, message):
        """
        Logs a message to the .log file.

        Args:
            message (str): Message to log.
        """
        self.file_logger.info(message)

    def log_to_csv(self, epoch, mode, loss, accuracy=None):
        """
        Logs metrics to the .csv file.

        Args:
            epoch (int): Epoch number.
            mode (str): Mode ("train" or "test").
            loss (float): Loss value.
            accuracy (float): Accuracy value (optional).
        """
        with open(self.csv_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([epoch, mode, loss, accuracy])

    def log_to_tensorboard(self, epoch, loss, accuracy=None, mode="train"):
        """
        Logs metrics to TensorBoard.

        Args:
            epoch (int): Epoch number.
            loss (float): Loss value.
            accuracy (float): Accuracy value (optional).
            mode (str): Mode ("train" or "test").
        """
        self.tensorboard_writer.add_scalar(f"loss/{mode}", loss, epoch)  # Shared prefix "loss/"
        if accuracy is not None:
            self.tensorboard_writer.add_scalar(f"accuracy/{mode}", accuracy, epoch)


    def log(self, epoch, mode, loss, accuracy=None):
        """
        Logs metrics to all logging targets (file, CSV, TensorBoard).

        Args:
            epoch (int): Epoch number.
            mode (str): Mode ("train" or "test").
            loss (float): Loss value.
            accuracy (float): Accuracy value (optional).
        """
        message = f"Epoch: {epoch}, Mode: {mode}, Loss: {loss:.6f}, Accuracy: {accuracy if accuracy is not None else 'N/A'}"
        self.log_to_file(message)
        self.log_to_csv(epoch, mode, loss, accuracy)
        self.log_to_tensorboard(epoch, loss, accuracy, mode)

    def close(self):
        """
        Closes the TensorBoard writer.
        """
        self.tensorboard_writer.close()
