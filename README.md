# **Decentralized and Centralized Federated Learning for Autonomous Vehicles**

## **Overview**

This project explores the performance comparison between **decentralized** and **centralized** federated learning approaches for autonomous vehicles. The vehicles collaboratively train a neural network to predict steering angles based on video frames and optical flow data. 

Key highlights of this project:
- Utilizes a **dual-stream convolutional neural network** (CNN) to process spatial and temporal information.
- Implements **federated learning** simulations for both centralized and decentralized architectures.
- Processes video data with **frame streams** and **optical flow streams** to learn steering behaviors.
- Evaluates the efficiency, accuracy, and scalability of centralized and decentralized learning.

---

## **Features**

1. **Dual-Stream Model**:
   - Frame stream processes three consecutive RGB frames.
   - Optical flow stream processes two optical flow maps derived from consecutive frames.

2. **Federated Learning**:
   - **Centralized**: A central server aggregates weights from all workers (vehicles).
   - **Decentralized**: Vehicles exchange weights directly with dynamically selected peers.

3. **Data Handling**:
   - Each vehicle gets a non-overlapping subset of the training data.
   - Ensures video frames and optical flow inputs remain contiguous for model training.

4. **Performance Comparison**:
   - Measures the effectiveness of decentralized learning versus centralized learning.
   - Evaluates the model on each worker's test dataset after training.

---

## **Project Structure**

```
Project/
├── data/
│   ├── base_model_training/
│   │   ├── data/
│   │   └── data.txt
│   ├── training_data/
│   │   ├── data/
│   │   └── data.txt
├── models/
│   └── base_model.py
├── simulations/
│   ├── centralized.py
│   ├── decentralized.py
├── utils/
│   ├── aggregation.py
│   ├── data_loader.py
│   └── split_dataset.py
├── train_base_model.py
└── README.md
```

---

## **Getting Started**

### **Prerequisites**
- Python 3.8 or higher
- PyTorch 2.0 or higher
- OpenCV for optical flow computation
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### **Dataset Structure**
The `data/` folder contains two datasets:
1. **Base Model Training Data**:
   - Folder: `data/base_model_training/`
   - File: `data/base_model_training/data.txt`
     - Format: `IMGNAME OUTANGLE`
     - Example:
       ```
       X_1.jpg 0.125
       X_2.jpg -0.75
       ```

2. **Worker Training Data**:
   - Folder: `data/training_data/`
   - File: `data/training_data/data.txt`
     - Includes timestamp information, but the model only uses the filename and steering angle.

---

### **Training Workflow**

#### **1. Train Base Model**
Train the global base model on the `base_model_training/` dataset:
```bash
python train_base_model.py \
    --data_folder "data/base_model_training/data/" \
    --data_file "data/base_model_training/data.txt" \
    --save_path "base_model.pth" \
    --epochs 10 \
    --batch_size 32 \
    --lr 0.0001 \
    --device "cuda"
```

#### **2. Centralized Federated Learning**
Simulate centralized federated learning with the trained base model:
```bash
python simulations/centralized.py \
    --data_folder "data/training_data/data/" \
    --data_file "data/training_data/data.txt" \
    --num_workers 5 \
    --rounds 5 \
    --epochs_per_worker 3 \
    --base_model_path "base_model.pth" \
    --device "cuda"
```

#### **3. Decentralized Federated Learning**
Simulate decentralized federated learning with dynamically selected neighbors:
```bash
python simulations/decentralized.py \
    --data_folder "data/training_data/data/" \
    --data_file "data/training_data/data.txt" \
    --num_workers 5 \
    --rounds 5 \
    --epochs_per_worker 3 \
    --base_model_path "base_model.pth" \
    --device "cuda"
```

---

## **Core Components**

### **Dual-Stream Model**
- **Frame Stream Input**:
  - Processes three consecutive RGB frames $\{ A_{t-2}, A_{t-1}, A_t \}$.
- **Optical Flow Stream Input**:
  - Processes two optical flow maps:
    - $O_{t-1} = f(A_{t-2}, A_{t-1})$
    - $O_t = f(A_{t-1}, A_t)$
- Predicts the steering angle for $A_t$.

### **Loss Function**
The loss is computed as:
```math
\text{Loss} = \frac{1}{N} \sum_{t=1}^N (\theta_t - \hat{\theta}_t)^2
```
Where:
- $\theta_t$: Ground truth steering angle for frame $A_t$.
- $\hat{\theta}_t$: Predicted steering angle for $A_t$.

### **Worker Implementation**
Each worker:
1. Receives a subset of the training data.
2. Trains locally using Mean Squared Error (MSE) loss.
3. Sends its model weights to the central server (centralized) or shares them with peers (decentralized).

### **Dynamic Neighbor Selection (Decentralized Learning)**
For each round, each worker randomly selects 2-4 peers as neighbors for weight aggregation. This simulates real-world dynamic connectivity between vehicles.

---

## **Key Results**
1. **Accuracy Comparison**:
   - Evaluate the model's test loss after centralized and decentralized training.
2. **Efficiency**:
   - Compare training times for centralized and decentralized simulations.
3. **Scalability**:
   - Assess the impact of the number of workers on performance.

---

## **Future Work**
- Incorporate real-world driving data for testing.
- Explore alternative communication protocols for decentralized learning.
- Test different neural network architectures.

---

## **Acknowledgments**
This project is inspired by research on federated learning for autonomous vehicles. The optical flow computation uses **Gunnar Farneback's algorithm** implemented in OpenCV.

---

## **License**
This project is licensed under the MIT License.
