# Intelligent Traffic Congestion Prediction and Control using Vision and GNN

A deep learning system for real-time urban traffic forecasting and congestion control, combining Computer Vision (YOLOv8) with an Enhanced Graph Neural Network (E-GNN) architecture.

---

## Overview

Urban traffic congestion is a growing challenge in smart city management. This project proposes an **Enhanced Graph Neural Network (E-GNN)** that models traffic networks as graphs — with intersections/sensors as nodes and road segments as edges — to deliver accurate, multi-horizon traffic speed and congestion predictions.

The system is validated on two real-world benchmark datasets: **PeMS-BAY** and **METR-LA**, achieving a **26.25% reduction in RMSE** at the 15-minute prediction horizon compared to state-of-the-art baselines.

---

## Key Features

- **E-GNN Architecture** — Dynamic adjacency learning, stacked GNN layers with gated temporal convolutions, and spatial-temporal attention mechanisms
- **Multimodal Input Fusion** — Combines traffic sensor data (speed, flow, occupancy) with temporal and weather features
- **YOLOv8 Vehicle Detection** — Real-time vehicle counting and road-segment density estimation from video frames
- **Multi-step Prediction** — Simultaneous forecasts at 15, 30, and 60-minute horizons
- **Flask Web Interface** — Manual slider-based prediction and AI video analysis in a single dashboard

---

## System Architecture

```
Traffic Video / Sensor Data
        │
        ▼
┌─────────────────┐     ┌──────────────────────┐
│  YOLOv8 Vision  │────▶│  Multimodal Preprocess│
│  (Vehicle Det.) │     │  (Speed, Weather, Time)│
└─────────────────┘     └──────────┬───────────┘
                                   │
                         ┌─────────▼──────────┐
                         │  E-GNN Layers       │
                         │  (Dynamic Adj. +    │
                         │   Attention + TCN)  │
                         └─────────┬──────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  Multi-step Decoder          │
                    │  (15 min / 30 min / 60 min)  │
                    └─────────────────────────────┘
                                   │
                         Congestion Level + Signal Recommendation
```

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.8+ |
| Deep Learning | PyTorch 1.12+, PyTorch Geometric 2.1 |
| Computer Vision | YOLOv8 (Ultralytics) |
| Data Processing | NumPy, Pandas, Scikit-learn |
| Web Framework | Flask 2.0+ |
| Visualization | Matplotlib, Seaborn |
| GPU Acceleration | CUDA 11.3 |

---

## Datasets

| Dataset | Sensors | Duration | Interval |
|---|---|---|---|
| **METR-LA** | 207 | 4 months (Mar–Jun 2012) | 5 min |
| **PeMS-BAY** | 325 | 6 months (Jan–Jun 2017) | 5 min |

Both datasets are publicly available via the [Caltrans PeMS portal](https://pems.dot.ca.gov/).

---

## Results

### METR-LA Benchmark

| Model | MAE (15 min) | RMSE (15 min) | MAPE (15 min) | MAE (60 min) | RMSE (60 min) |
|---|---|---|---|---|---|
| ARIMA | 3.99 | 8.21 | 9.60% | — | — |
| FC-LSTM | 3.44 | 6.30 | 9.60% | 3.77 | 6.89 |
| DCRNN | 2.77 | 5.38 | 7.30% | 3.15 | 6.45 |
| Graph WaveNet | 2.69 | 5.15 | 6.90% | 3.53 | 7.37 |
| ASTGCN | 2.72 | 5.28 | 7.01% | 3.40 | 7.02 |
| GMAN | 2.80 | 5.55 | 7.41% | 3.44 | 7.21 |
| **E-GNN (Ours)** | **2.41** | **3.80** | **6.12%** | **2.89** | **5.96** |

### Ablation Study

| Variant | MAE (15 min) | RMSE (15 min) |
|---|---|---|
| E-GNN (Full) | 2.41 | 3.80 |
| w/o Dynamic Adjacency | 2.62 | 4.12 |
| w/o Multimodal Input | 2.55 | 4.01 |
| w/o Attention | 2.73 | 4.38 |
| w/o Temporal Conv | 2.89 | 4.67 |

---

## Hardware Requirements

| Component | Specification |
|---|---|
| Processor | Intel Core i7 / AMD Ryzen 7 or higher |
| RAM | 16 GB DDR4 (32 GB recommended for training) |
| GPU | NVIDIA GTX 1080 Ti / RTX 3060 (8 GB VRAM min) |
| Storage | 256 GB SSD + 1 TB HDD |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/intelligent-traffic-egnn.git
cd intelligent-traffic-egnn

# Create a virtual environment
conda create -n traffic_egnn python=3.9
conda activate traffic_egnn

# Install PyTorch with CUDA
pip install torch==1.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# Install PyTorch Geometric
pip install torch-geometric==2.1.0

# Install remaining dependencies
pip install -r requirements.txt
```

---

## Usage

### Run the Flask Web App

```bash
python app.py
```

Navigate to `http://localhost:5000` to access the dashboard.

**Manual Prediction:** Select an intersection node and adjust the 4 route density sliders to get a congestion prediction (LOW / MEDIUM / HIGH).

**AI Video Analysis:** Upload a traffic video, capture a frame, and run the E-GNN + YOLOv8 analysis to get real-time congestion level and green-light recommendations.

---

## Model Training

```bash
python train.py --dataset METR-LA --epochs 200 --batch_size 64 --lr 0.001
```

Key hyperparameters:

| Parameter | Value |
|---|---|
| Input window | 12 time steps (60 min) |
| Hidden dimension | 64 |
| E-GNN layers | 3 |
| TCN dilation factors | [1, 2, 4, 8] |
| Optimizer | Adam (lr=0.001) |
| Loss | Masked MAE |

---

## Project Structure

```
intelligent-traffic-egnn/
├── app.py                  # Flask web application
├── train.py                # Model training script
├── models/
│   ├── egnn.py             # E-GNN architecture
│   └── traffic_gnn.py      # GNN classifier for slider prediction
├── data/
│   ├── metr-la/
│   └── pems-bay/
├── static/
│   └── uploads/            # Saved output frames
├── templates/
│   └── index.html          # Web dashboard
├── traffic_gnn_model.pth   # Pre-trained GNN weights
├── yolov8l.pt              # YOLOv8 large weights
└── requirements.txt
```

---

## Future Work

- Replace TCN blocks with Transformer encoders for better long-range temporal modeling
- Federated learning across distributed Traffic Management Centers
- Reinforcement learning agent for adaptive traffic signal control
- Edge deployment via model pruning and quantization
- Multi-city transfer learning for cross-domain generalization

---

## References

- Li et al., *DCRNN: Data-driven traffic forecasting*, ICLR 2018
- Wu et al., *Graph WaveNet*, IJCAI 2019
- Zheng et al., *GMAN*, AAAI 2020
- Guo et al., *ASTGCN*, AAAI 2019
- Kipf & Welling, *GCN*, ICLR 2017

---

## Author

**Turubhatla Naga Sumanth** (24L31F00I6)  
Masters in Computer Applications  
Vignan's Institute of Information Technology (Autonomous), Visakhapatnam  
*Under the guidance of Dr. B. Prasad, Assistant Professor, Dept. of MCA*
