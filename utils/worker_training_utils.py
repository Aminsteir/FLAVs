from concurrent.futures import ProcessPoolExecutor
from functools import partial


def train_worker(worker, epochs, lr, subset_ratio):
    """
    Train a single worker and return its weights and loss.

    Args:
        worker: The worker object.
        epochs: Number of epochs to train.
        lr: Learning rate.
        subset_ratio: Fraction of the dataset to use for training.

    Returns:
        dict: Worker ID, trained weights, and average loss.
    """
    avg_loss = worker.train(epochs=epochs, lr=lr, subset_ratio=subset_ratio)
    weights = worker.send_weights()
    return {"worker_id": worker.worker_id, "weights": weights, "avg_loss": avg_loss}


def evaluate_worker(worker):
    """
    Evaluate a single worker and return its average loss.

    Args:
        worker: The worker object.

    Returns:
        dict: Worker ID and average loss.
    """
    avg_loss = worker.evaluate()
    return {"worker_id": worker.worker_id, "avg_loss": avg_loss}


def parallelize_workers(workers, train=True, epochs=1, lr=1e-5, subset_ratio=1.0):
    """
    Parallelize worker training or evaluation.

    Args:
        workers (list): List of worker objects.
        train (bool): Whether to train or evaluate the workers.
        epochs (int): Number of training epochs (if train=True).
        lr (float): Learning rate (if train=True).
        subset_ratio (float): Subset ratio for training (if train=True).

    Returns:
        list[dict]: Ordered results from each worker.
    """
    func = train_worker if train else evaluate_worker
    if train:
        func = partial(func, epochs=epochs, lr=lr, subset_ratio=subset_ratio)

    results = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(func, worker) for worker in workers]
        for future in futures:
            results.append(future.result())

    # Ensure the results are ordered by worker ID
    results.sort(key=lambda x: x["worker_id"])
    return results
