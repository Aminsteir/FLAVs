import torch

def federated_average(weights_list):
    """
    Performs federated averaging of model weights.
    
    Args:
        weights_list (list[dict]): List of model weight dictionaries from workers.
    
    Returns:
        dict: Averaged model weights.
    """
    # Initialize an empty dictionary to store averaged weights
    avg_weights = {}

    # Iterate through each layer/key in the weights
    for key in weights_list[0].keys():
        # Stack weights from all workers for the given key
        stacked_weights = torch.stack([weights[key] for weights in weights_list])
        # Compute the average of these weights
        avg_weights[key] = torch.mean(stacked_weights, dim=0)

    return avg_weights

def weighted_federated_average(weights_list, weights_proportions):
    """
    Performs weighted federated averaging of model weights.
    
    Args:
        weights_list (list[dict]): List of model weight dictionaries from workers.
        weights_proportions (list[float]): List of weights (proportions) for each worker, summing to 1.
    
    Returns:
        dict: Weighted averaged model weights.
    """
    # Ensure proportions sum to 1
    if not torch.isclose(torch.tensor(weights_proportions).sum(), torch.tensor(1.0)):
        raise ValueError("Weights proportions must sum to 1.")

    # Initialize an empty dictionary to store averaged weights
    avg_weights = {}

    # Iterate through each layer/key in the weights
    for key in weights_list[0].keys():
        # Perform weighted sum of weights for the given key
        weighted_sum = sum(w[key] * proportion for w, proportion in zip(weights_list, weights_proportions))
        avg_weights[key] = weighted_sum

    return avg_weights
