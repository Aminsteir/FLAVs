import random
from torch.utils.data import DataLoader

def swap_worker_data(workers):
    """
    Randomly swap training data between workers.

    Args:
        workers (list): List of Worker objects.
    """
    datasets = [worker.train_dataset for worker in workers]
    random.shuffle(datasets)
    for i, worker in enumerate(workers):
        worker.train_dataset = datasets[i]