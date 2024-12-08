# **Autonomous Vehicle Federated Learning**

## **Overview**

This project investigates the application of **federated learning** (FL) to autonomous driving systems, comparing the **centralized** and **decentralized** FL architectures. The primary objective is to collaboratively train neural networks across edge devices (vehicles) to predict steering angles from video data while ensuring data privacy and scalability.

### **Goals**
1. Train autonomous vehicle models to predict steering angles effectively using video frames and optical flow (if using **Dual-Stream Model**).
2. Compare the performance of **centralized FL** and **decentralized FL** in terms of:
   - Test accuracy.
   - Training efficiency.
   - Scalability.

---

## **Features**

1. **Model Variants**:
   - **Dual-Stream Model**: Processes frame and optical flow streams separately.
   - **Spatio-Temporal Model**: Incorporates temporal relationships with 3D convolutions.

2. ** Output Handling**:
   - Models currently predict raw output steering angle, but we have plans to integrate different output types:
     - `sin_cos`: Predicts `sin` and `cos` components of angles.

3. **Federated Learning**:
   - **Centralized FL**: A central server aggregates model weights from vehicles.
   - **Decentralized FL**: Vehicles directly exchange weights with dynamic peers.

4. **Visualization Tools**:
   - Generate videos overlaying predicted and ground truth steering angles.
   - Analyze inference time to ensure real-time feasibility.

5. **Worker Simulation**:
   - Simulates edge devices (vehicles) by splitting datasets across workers.
   - Supports dynamic peer selection for decentralized learning.

---

## **Project Structure**

```
Project/
├── data/                          # Datasets for training and testing
│   ├── base_model_training/       # Data for training the initial base model
│   ├── training_data/             # Data for federated learning simulations
│
├── models/                        # Neural network architectures
│   ├── dual_stream.py             # Dual-Stream Model
│   ├── spatio_temporal.py         # Spatio-Temporal Model
│   ├── registry.py                # Model registry for dynamic retrieval
│
├── scheduler/                     # Schedule training jobs
│   ├── base_model_training.py     # Scheduler to train all the base models
│   ├── fl_training.py             # Scheduler to train all the federated learning models
│
├── simulations/                   # Federated learning simulation scripts
│   ├── centralized.py             # Centralized FL simulation
│   ├── decentralized.py           # Decentralized FL simulation
│   ├── worker.py                  # Individual worker class for FL
│
├── utils/*                        # Utility functions used for training / testing
├── train_base_model.py            # Script for training the base model
│
├── testing/                       # Visualization tools
│   ├── fl_testing_analysis.py     # Test and evaluate the federated learning models
│   ├── model_performance_video.py # Generate video overlays for predictions
│   ├── model_viz.py               # Generate and view model architecture (img format)
│   ├── plot_steering_angles.py    # Generate model prediction plots compared to ground truth
│   ├── visualize_frames.py        # View sample optical flow images that are used in training / test
│
└── README.md                      # Project documentation
```

---

## **Training Workflow**

### **1. Train Base Model**
The base model is trained on the initial dataset using the desired output type and loss function. This serves as the foundation for federated learning.
```bash
python train_base_model.py \
    --model_type "dual_stream" \
    --output_type "sin_cos" \
    --data_folder "data/base_model_training/data/" \
    --data_file "data/base_model_training/data.txt" \
    --save_dir "build/" \
    --epochs 10 \
    --batch_size 32 \
    --lr 0.001 \
    --device "cuda"
```

### **2. Centralized Federated Learning**
Simulate centralized FL where a central server aggregates model weights from workers.
```bash
python simulations/centralized.py \
    --model_type "dual_stream" \
    --output_type "sin_cos" \
    --data_folder "data/training_data/data/" \
    --data_file "data/training_data/data.txt" \
    --save_dir "build/" \
    --num_workers 5 \
    --rounds 5 \
    --epochs_per_worker 3 \
    --batch_size 8 \
    --base_model_path "build/dual_stream-sin_cos-base_model.pth" \
    --device "cuda"
```

### **3. Decentralized Federated Learning**
Simulate decentralized FL where workers exchange weights with dynamically selected peers.
```bash
python simulations/decentralized.py \
    --model_type "dual_stream" \
    --output_type "sin_cos" \
    --data_folder "data/training_data/data/" \
    --data_file "data/training_data/data.txt" \
    --save_dir "build/" \
    --num_workers 5 \
    --rounds 5 \
    --epochs_per_worker 3 \
    --batch_size 8 \
    --base_model_path "build/dual_stream-sin_cos-base_model.pth" \
    --device "cuda"
```

### **4. Visualization**
Generate videos to compare ground truth and predicted steering angles.
```bash
python visualization/model_performance_video.py \
    --model_path "build/dual_stream-sin_cos-base_model.pth" \
    --model_type "dual_stream" \
    --output_type "sin_cos" \
    --data_folder "data/training_data/data/" \
    --data_file "data/training_data/data.txt" \
    --subset_fraction 0.02 \
    --output_dir "visualizations/" \
    --fps 30 \
    --device "cuda"
```

---

## **Key Components**

### **Models**
- **Dual-Stream Model**: Separately processes RGB frames and optical flow using convolutional layers.
- **Spatio-Temporal Model**: Combines spatial and temporal information with 3D convolutions.

### **Output Representation**
- `angle`: Steering angle in degrees.
- `sin_cos` (Future): Predicts sine and cosine of angles for better gradient behavior.

### **Federated Learning**
1. **Centralized FL**:
   - Server aggregates weights from all workers.
   - Simple but requires consistent connectivity to a central server.
2. **Decentralized FL**:
   - Workers exchange weights with peers.
   - More robust to network failures.

### **Visualization**
- Draws overlaid steering wheels to compare predictions against ground truth.
- Computes inference time to ensure real-time suitability.

---

## **Results**

1. **Inference**:
   - All models achieved inference times suitable for real-time deployment (FPS > 30).

---

## **Future Work**
- Incorporate real-world datasets for better generalization.
- Train the models in a simulated test environment.
- Explore additional neural architectures (e.g., Vision Transformers).
- Optimize decentralized weight-sharing protocols for scalability.

---

## **License**
This project is licensed under the MIT License.
