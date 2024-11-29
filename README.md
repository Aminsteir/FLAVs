# **Federated Learning for Autonomous Vehicle Steering Prediction**

## **Overview**

This project investigates the application of **federated learning** to autonomous vehicles, focusing on predicting steering angles from video data. The research compares the effectiveness of **centralized** and **decentralized** federated learning paradigms, simulating vehicle-to-vehicle communication for model updates.

Key aspects include:
- Implementation of multiple **neural network architectures**, such as **dual-stream CNNs** and **temporal transformers**.
- Flexible configurations using a **ModelConfig** class to switch between various output formats (e.g., `angle`, `sin_cos`) and loss functions.
- Robust handling of spatial and temporal data using **frame sequences** and **optical flow**.
- Simulation of dynamic peer-to-peer communication for decentralized learning.

---

## **Features**

1. **Model Architectures**:
   - **Dual-Stream CNN**: Processes spatial (frame stream) and temporal (optical flow) information.
   - **Spatio-Temporal CNN**: Incorporates 3D convolutions for spatio-temporal processing.
   - **Temporal Transformer**: Uses transformers to process temporal sequences for steering angle prediction.

2. **Federated Learning Approaches**:
   - **Centralized**: A global server aggregates model updates from all workers.
   - **Decentralized**: Vehicles dynamically select peers for weight aggregation.

3. **Dynamic Output Configurations**:
   - Predict steering angles directly (`angle`).
   - Predict normalized angles (`angle_norm`) for better training stability.
   - Predict `sin` and `cos` of the angle (`sin_cos`) for circular loss handling.

4. **Flexible Data Handling**:
   - Seamlessly integrates frame sequences and optical flow.
   - Dynamically prepares targets based on the selected output type.

---

## **Project Structure**

```
.
├── LICENSE
├── README.md
├── build/                     # Contains trained model weights
│   ├── base_model.pth
│   └── tt_base_model.pth
├── data/                      # Dataset-related files
│   ├── base_model_training/   # Data for initial base model training
│   ├── original_data.zip      # Original dataset archive
│   └── training_data/         # Data distributed to workers
├── logs/                      # Training logs for different scenarios
├── models/                    # Model architectures and configuration
│   ├── base_model.py
│   ├── dual_stream.py
│   ├── model_config.py
│   ├── spatio_temporal.py
│   └── temporal_transformer.py
├── simulations/               # Centralized and decentralized FL simulations
├── testing/                   # Testing and visualization scripts
├── train_base_model.py        # Script to train the initial base model
├── utils/                     # Helper functions for data processing and logging
└── workers/                   # Worker implementation for federated learning
```

---

## **Getting Started**

### **Prerequisites**
- Python 3.8+
- PyTorch 2.0+
- OpenCV for optical flow computation

### **Dataset Structure**
1. **Base Model Training Data** (`data/base_model_training/`):
   - `data.txt` contains mappings: `IMGNAME OUTANGLE`
     ```
     X_1.jpg 15.3
     X_2.jpg -7.8
     ```
2. **Worker Training Data** (`data/training_data/`):
   - Similar to the base training data but distributed to workers.

---

## **Workflow**

### **1. Train the Base Model**
Train a global model using the base dataset:
```bash
python train_base_model.py \
    --model_type "temporal_transformer" \
    --data_folder "data/base_model_training/data/" \
    --data_file "data/base_model_training/data.txt" \
    --save_path "build/base_model.pth" \
    --epochs 10 \
    --batch_size 32 \
    --lr 0.0001 \
    --device "cuda"
```

### **2. Simulate Federated Learning**

#### **Centralized**
Simulate centralized federated learning:
```bash
python simulations/centralized.py \
    --data_folder "data/training_data/data/" \
    --data_file "data/training_data/data.txt" \
    --num_workers 5 \
    --rounds 5 \
    --epochs_per_worker 3 \
    --base_model_path "build/base_model.pth" \
    --device "cuda"
```

#### **Decentralized**
Simulate decentralized federated learning with dynamic neighbor selection:
```bash
python simulations/decentralized.py \
    --data_folder "data/training_data/data/" \
    --data_file "data/training_data/data.txt" \
    --num_workers 5 \
    --rounds 5 \
    --epochs_per_worker 3 \
    --base_model_path "build/base_model.pth" \
    --device "cuda"
```

---

## **Key Components**

### **1. Dynamic Model Configurations**
The `ModelConfig` class allows for easy switching between model types and output formats:
- Output types: `angle`, `angle_norm`, `sin_cos`
- Loss functions: `mse_loss`, `circular_loss`

### **2. Federated Learning Framework**
- **Centralized**: Workers send weights to a server for aggregation.
- **Decentralized**: Workers dynamically exchange weights with peers.

### **3. Dynamic Neighbor Selection**
- Each round, workers randomly select 2-4 peers for weight aggregation.

---

## **Visualization**

Generate a video comparing model predictions to ground truth steering angles:
```bash
python testing/model_performance_video.py \
    --model_path "build/tt_base_model.pth" \
    --model_type "temporal_transformer" \
    --data_folder "data/training_data/data/" \
    --data_file "data/training_data/data.txt" \
    --subset_fraction 0.02 \
    --output_video "performance_video.mp4" \
    --device "cuda"
```

---

## **Future Directions**
- Use real-world driving data for validation.
- Implement alternative aggregation strategies for decentralized learning.
- Explore lightweight neural architectures for real-time inference.

---

## **Acknowledgments**
This project leverages **PyTorch** for neural network training and **OpenCV** for optical flow computation. Inspired by federated learning research in autonomous driving.

---

## **License**
This project is licensed under the MIT License.
