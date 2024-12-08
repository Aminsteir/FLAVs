import math
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

    logs = [
        (4, load_csv_logs(CENTRALIZED_4_LOG), load_csv_logs(DECENTRALIZED_4_LOG)),
        (8, load_csv_logs(CENTRALIZED_8_LOG), load_csv_logs(DECENTRALIZED_8_LOG))
    ]

    for log_t in logs:
        n, c_n, d_n = log_t
        for mode in ["train", "test"]:
            plt.figure(figsize=(10,6))
            plt.plot(c_n[c_n["Mode"]==mode]["Epoch"], c_n[c_n["Mode"]==mode]["Loss"], label=f"Centralized {n}W - {mode.title()}")
            plt.plot(d_n[d_n["Mode"]==mode]["Epoch"], d_n[d_n["Mode"]==mode]["Loss"], label=f"Decentralized {n}W - {mode.title()}")

            plt.xlabel("Round")
            plt.ylabel(f"Avg. {mode.title()} Loss")
            plt.title(f"Avg. {mode.title()} Loss over Federated Rounds")
            plt.grid(True)
            plt.legend()
            plt.savefig(f"testing/fl_analysis_outputs/{n}W_{mode}_loss_comparison.png")
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
    Plot ground truth vs predicted.
    model_dict: { "Scenario": model }
    """
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
    plt.savefig("testing/fl_analysis_outputs/gt_vs_pred_overlayed.png")
    plt.close()


def plot_models_subplots(model_dict, dataset, device=DEVICE, output_path="testing/fl_analysis_outputs/scenarios_subplots.png"):
    """
    Similar to the above function, but instead of overlaying all models in one plot,
    we create a subplot for each model scenario and include ground truth in each subplot.

    Args:
        model_dict (dict): { "Scenario": model }
        dataset: The dataset to evaluate.
        device: The device for running models.
        output_path: File path to save the resulting plot.
    """
    indices = list(range(len(dataset)))

    # Collect ground truth once
    ground_truth = []
    for idx in indices:
        *_, target = dataset[idx]
        ground_truth.append(target.item())

    scenarios = list(model_dict.keys())
    num_scenarios = len(scenarios)

    # Determine subplot grid size
    ncols = math.ceil(math.sqrt(num_scenarios))
    nrows = math.ceil(num_scenarios / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(nrows, ncols)

    for i, scenario in enumerate(scenarios):
        model = model_dict[scenario]
        predictions = []
        model.eval()
        with torch.no_grad():
            for idx in indices:
                *inputs, target = dataset[idx]
                inputs = [inp.unsqueeze(0).to(device) for inp in inputs]
                pred = model(*inputs).item()
                predictions.append(pred)

        r = i // ncols
        c = i % ncols
        ax = axes[r, c]

        ax.plot(indices, ground_truth, label="Ground Truth", color='black', linewidth=2)
        ax.plot(indices, predictions, linestyle='--', label=scenario)

        ax.set_title(scenario)
        ax.set_xlabel("Frame Index")
        ax.set_ylabel("Steering Angle (degrees)")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_decentralized_workers(worker_models, dataset, device=DEVICE, output_path="testing/fl_analysis_outputs/decentr_workers_pred.png"):
    """
    Plot each decentralized worker model predictions in its own subplot with ground truth.

    Args:
        worker_models (list or dict): If a list, each element is a model. If a dict, keys are worker IDs and values are models.
        dataset: The dataset to evaluate.
        device: The device for running models.
        output_path: File path to save the resulting plot.
    """
    # Convert worker_models to a list if it's a dict
    if isinstance(worker_models, dict):
        worker_models_list = [(k, v) for k,v in worker_models.items()]
    else:
        # If it's just a list, create dummy worker IDs
        worker_models_list = [(f"Worker {i}", m) for i, m in enumerate(worker_models)]

    num_workers = len(worker_models_list)
    indices = list(range(len(dataset)))

    # Collect ground truth once
    ground_truth = []
    for idx in indices:
        *_, target = dataset[idx]
        ground_truth.append(target.item())

    # Determine subplot grid size
    ncols = math.ceil(math.sqrt(num_workers))
    nrows = math.ceil(num_workers / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(nrows, ncols)

    for i, (worker_id, model) in enumerate(worker_models_list):
        # Compute predictions for this worker
        predictions = []
        model.eval()
        with torch.no_grad():
            for idx in indices:
                *inputs, target = dataset[idx]
                inputs = [inp.unsqueeze(0).to(device) for inp in inputs]
                pred = model(*inputs).item()
                predictions.append(pred)

        r = i // ncols
        c = i % ncols
        ax = axes[r, c]

        ax.plot(indices, ground_truth, label="Ground Truth", color='black', linewidth=2)
        ax.plot(indices, predictions, linestyle='--', label=worker_id)

        ax.set_title(f"{worker_id}")
        ax.set_xlabel("Frame Index")
        ax.set_ylabel("Steering Angle (degrees)")
        ax.grid(True)
        ax.legend()

    # Adjust layout to prevent overlapping labels
    plt.tight_layout()
    plt.savefig(output_path)
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

    # Previously, we plotted multiple scenarios on one graph. Now let's use subplots:
    # Create a scenario dictionary for centralized/decentralized global models:
    model_dict_for_plot = {
        "Centr.4W": c4_model,
        "Centr.8W": c8_model,
        "Decentr.4W Global": d4_global_model,
        "Decentr.8W Global": d8_global_model
    }

    plot_ground_truth_vs_pred(model_dict_for_plot, test_dataset)

    # Plot each scenario in its own subplot
    plot_models_subplots(model_dict_for_plot, test_dataset, device=DEVICE, 
                         output_path="testing/fl_analysis_outputs/scenarios_subplots.png")

    # Also, plot each decentralized worker model in its own subplot
    # For clarity, let's make a dictionary for the 4-worker setup and 8-worker setup
    d4_worker_dict = {f"Worker {i}": d4_worker_models[i] for i in range(len(d4_worker_models))}
    d8_worker_dict = {f"Worker {i}": d8_worker_models[i] for i in range(len(d8_worker_models))}

    plot_decentralized_workers(d4_worker_dict, test_dataset, device=DEVICE, 
                               output_path="testing/fl_analysis_outputs/decentr_4workers_subplots.png")

    plot_decentralized_workers(d8_worker_dict, test_dataset, device=DEVICE,
                               output_path="testing/fl_analysis_outputs/decentr_8workers_subplots.png")

    print("Plots saved in 'testing/fl_analysis_outputs/' directory.")

if __name__ == "__main__":
    main()
