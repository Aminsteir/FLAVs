import torch

def split_dataset_for_workers(dataset, num_workers, sequence_length=3):
    """
    Split the dataset into contiguous subsets for each worker.

    Args:
        dataset (Dataset): Complete dataset to be split.
        num_workers (int): Number of workers.
        sequence_length (int): Number of consecutive frames required by the model.

    Returns:
        list[Dataset]: List of datasets, one for each worker.
    """
    dataset_size = len(dataset)

    # Ensure there is enough data for each worker to have at least one sequence
    if dataset_size < num_workers * sequence_length:
        raise ValueError(
            "Not enough data to split among workers with the given sequence length."
        )

    # Compute the size of each worker's dataset
    worker_data_sizes = [dataset_size // num_workers] * num_workers
    for i in range(dataset_size % num_workers):
        worker_data_sizes[i] += 1

    # Make sure each worker's dataset size is sufficient for at least one sequence
    worker_data_sizes = [max(size, sequence_length) for size in worker_data_sizes]

    # Generate start indices for each worker's contiguous subset
    start_indices = [sum(worker_data_sizes[:i]) for i in range(num_workers)]

    # Create contiguous subsets for each worker
    subsets = [
        torch.utils.data.Subset(dataset, list(range(start, start + size)))
        for start, size in zip(start_indices, worker_data_sizes)
    ]

    return subsets