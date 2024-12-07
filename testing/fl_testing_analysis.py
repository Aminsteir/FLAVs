import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utils.data_loader import AutonomousVehicleDataset
from models.registry import get_model
from utils.aggregation import federated_average
from sklearn.metrics import mean_squared_error

# CONFIGURE YOUR PATHS AND SETTINGS
MODEL_TYPE = "dual_stream"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths for centralized final models
CENTRALIZED_4_PATH = "build/centralized/4_workers/dual_stream-centralized-finished-global.pth"
CENTRALIZED_8_PATH = "build/centralized/8_workers/dual_stream-centralized-finished-global.pth"

# Paths for decentralized final models (assume we have 4 and 8 workers)
DECENTRALIZED_4_PATHS = [
    f"build/decentralized/4_workers/dual_stream-decentralized-finished-worker_{i}.pth" for i in range(4)
]
DECENTRALIZED_8_PATHS = [
    f"build/decentralized/8_workers/dual_stream-decentralized-finished-worker_{i}.pth" for i in range(8)
]

# Logs for training curves
CENTRALIZED_4_LOG = "logs/centralized_4/centralized_4.csv"
CENTRALIZED_8_LOG = "logs/centralized_8/centralized_8.csv"
DECENTRALIZED_4_LOG = "logs/decentralized_4/decentralized_4.csv"
DECENTRALIZED_8_LOG = "logs/decentralized_8/decentralized_8.csv"

# Test dataset (use a common test dataset that none of the workers trained on)
# Adjust these paths as appropriate
TEST_DATA_FOLDER = "data/training_data/data/"
TEST_DATA_FILE = "data/training_data/data.txt"
TEST_SUBSET_FRACTION = 0.1  # Use a small subset for evaluation and plotting

#############################################
# Helper functions
#############################################

def load_model(model_path, model_type=MODEL_TYPE, device=DEVICE):
    model = get_model(model_type).to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def compute_rmse(model, dataset, device=DEVICE):
    model.eval()
    ground_truths = []
    predictions = []
    with torch.no_grad():
        for item in dataset:
            *inputs, target = item
            inputs = [inp.unsqueeze(0).to(device) for inp in inputs]
            pred = model(*inputs).item()
            predictions.append(pred)
            ground_truths.append(target.item())
    return np.sqrt(mean_squared_error(ground_truths, predictions))

def aggregate_weights(model_paths):
    """Aggregate weights from multiple workers using federated averaging."""
    weights_list = []
    for path in model_paths:
        weights_list.append(torch.load(path, map_location='cpu', weights_only=True))
    avg_weights = federated_average(weights_list)
    return avg_weights

def load_csv_logs(log_path):
    """Load CSV logs that have columns: Epoch, Mode, Loss, Accuracy."""
    df = pd.read_csv(log_path)
    # We assume that for federated setting "Epoch" corresponds to round
    return df

def plot_training_curves():
    """Plot training curves for all 4 scenarios."""
    # Load logs
    c4 = load_csv_logs(CENTRALIZED_4_LOG)
    c8 = load_csv_logs(CENTRALIZED_8_LOG)
    d4 = load_csv_logs(DECENTRALIZED_4_LOG)
    d8 = load_csv_logs(DECENTRALIZED_8_LOG)

    # Plot train loss over rounds
    plt.figure(figsize=(10,6))
    plt.plot(c4[c4["Mode"]=="train"]["Epoch"], c4[c4["Mode"]=="train"]["Loss"], label="Centralized 4W - Train")
    plt.plot(c8[c8["Mode"]=="train"]["Epoch"], c8[c8["Mode"]=="train"]["Loss"], label="Centralized 8W - Train")
    plt.plot(d4[d4["Mode"]=="train"]["Epoch"], d4[d4["Mode"]=="train"]["Loss"], label="Decentralized 4W - Train")
    plt.plot(d8[d8["Mode"]=="train"]["Epoch"], d8[d8["Mode"]=="train"]["Loss"], label="Decentralized 8W - Train")

    plt.xlabel("Round")
    plt.ylabel("Training Loss")
    plt.title("Training Loss over Federated Rounds")
    plt.grid(True)
    plt.legend()
    plt.savefig("testing/fl_analysis_outputs/training_loss_comparison.png")
    plt.close()

def plot_rmse_bar(rmses_dict):
    """
    Plot a bar chart of RMSE values.
    rmses_dict: { "Scenario": rmse_value }
    """
    scenarios = list(rmses_dict.keys())
    values = [rmses_dict[s] for s in scenarios]

    plt.figure(figsize=(8,5))
    plt.bar(scenarios, values, color=['blue','blue','red','red','green','green'])
    plt.ylabel("RMSE")
    plt.title("RMSE Comparison Across Configurations")
    plt.grid(axis='y')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("testing/fl_analysis_outputs/rmse_comparison.png")
    plt.close()

def plot_ground_truth_vs_pred(model_dict, dataset):
    """
    Plot ground truth vs predicted for a sample of frames.
    model_dict: { "Scenario": model }
    """
    # Take a sample from the dataset
    indices = list(range(len(dataset)))

    plt.figure(figsize=(10,6))
    # Ground truth
    ground_truth = []
    for idx in indices:
        *_, target = dataset[idx]
        ground_truth.append(target.item())

    plt.plot(indices, ground_truth, label="Ground Truth", color='black', linewidth=2)

    for scenario, model in model_dict.items():
        predictions = []
        with torch.no_grad():
            for idx in indices:
                *inputs, target = dataset[idx]
                inputs = [inp.unsqueeze(0).to(DEVICE) for inp in inputs]
                pred = model(*inputs).item()
                predictions.append(pred)

        plt.plot(indices, predictions, linestyle='--', label=scenario)

    plt.xlabel("Frame Index (Sample)")
    plt.ylabel("Steering Angle (degrees)")
    plt.title("Ground Truth vs Predicted Steering Angles")
    plt.grid(True)
    plt.legend()
    plt.savefig("testing/fl_analysis_outputs/gt_vs_pred.png")
    plt.close()

#############################################
# Main Analysis
#############################################
def main():
    os.makedirs("testing/fl_analysis_outputs/", exist_ok=True)

    # Load test dataset
    full_test_dataset = AutonomousVehicleDataset(
        data_folder=TEST_DATA_FOLDER, 
        data_file=TEST_DATA_FILE, 
        model_type=MODEL_TYPE
    )
    # We use a subset to speed up evaluation and plotting
    test_dataset = full_test_dataset.sample_subset(TEST_SUBSET_FRACTION)

    # Load centralized global models
    c4_model = load_model(CENTRALIZED_4_PATH)
    c8_model = load_model(CENTRALIZED_8_PATH)

    # Compute RMSE for centralized
    c4_rmse = compute_rmse(c4_model, test_dataset)
    c8_rmse = compute_rmse(c8_model, test_dataset)

    # Load decentralized worker models
    d4_worker_models = [load_model(p) for p in DECENTRALIZED_4_PATHS]
    d8_worker_models = [load_model(p) for p in DECENTRALIZED_8_PATHS]

    # Compute per-worker RMSE for decentralized
    d4_worker_rmses = [compute_rmse(m, test_dataset) for m in d4_worker_models]
    d8_worker_rmses = [compute_rmse(m, test_dataset) for m in d8_worker_models]

    # Average worker RMSE for decentralized
    d4_avg_rmse = np.mean(d4_worker_rmses)
    d8_avg_rmse = np.mean(d8_worker_rmses)

    # Also create an aggregated global model for decentralized to compare directly
    d4_agg_weights = aggregate_weights(DECENTRALIZED_4_PATHS)
    d8_agg_weights = aggregate_weights(DECENTRALIZED_8_PATHS)

    d4_global_model = get_model(MODEL_TYPE).to(DEVICE)
    d4_global_model.load_state_dict(d4_agg_weights)
    d4_global_rmse = compute_rmse(d4_global_model, test_dataset)

    d8_global_model = get_model(MODEL_TYPE).to(DEVICE)
    d8_global_model.load_state_dict(d8_agg_weights)
    d8_global_rmse = compute_rmse(d8_global_model, test_dataset)

    # Print out RMSE results
    print("=== RMSE Results ===")
    print(f"Centralized (4 workers): {c4_rmse:.4f}")
    print(f"Centralized (8 workers): {c8_rmse:.4f}")
    print(f"Decentralized (4 workers) - Avg. per-worker: {d4_avg_rmse:.4f}")
    print(f"Decentralized (4 workers) - Aggregated Global: {d4_global_rmse:.4f}")
    print(f"Decentralized (8 workers) - Avg. per-worker: {d8_avg_rmse:.4f}")
    print(f"Decentralized (8 workers) - Aggregated Global: {d8_global_rmse:.4f}")

    # Plot training curves (loss vs rounds)
    plot_training_curves()

    # Plot RMSE comparison as a bar chart
    rmses_to_plot = {
        "Centr. 4W Global": c4_rmse,
        "Centr. 8W Global": c8_rmse,
        "Decentr. 4W Avg": d4_avg_rmse,
        "Decentr. 4W Global": d4_global_rmse,
        "Decentr. 8W Avg": d8_avg_rmse,
        "Decentr. 8W Global": d8_global_rmse
    }
    plot_rmse_bar(rmses_to_plot)

    # Plot ground truth vs predictions for a subset of frames
    # We'll show lines for:
    # Centr. 4W Global, Centr. 8W Global,
    # Decentr. 4W Global (aggregated), Decentr. 8W Global (aggregated)
    model_dict_for_plot = {
        "Centr.4W": c4_model,
        "Centr.8W": c8_model,
        "Decentr.4W Global": d4_global_model,
        "Decentr.8W Global": d8_global_model
    }
    plot_ground_truth_vs_pred(model_dict_for_plot, test_dataset)

    print("Plots saved in 'testing/fl_analysis_outputs/' directory.")

if __name__ == "__main__":
    main()
