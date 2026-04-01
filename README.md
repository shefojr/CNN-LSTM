# Hand Gesture Recognition for Smart Home Control

## Project Overview
This repository contains my implementation of a **CNN-LSTM Neural Network** designed to recognize hand gestures from image sequences. This system is intended for **Smart Home Control**, providing a touchless interface for managing home devices (e.g., swiping to change lights, "OK" sign to confirm actions).

Recognizing gestures is a spatio-temporal challenge. It requires understanding:
1. **Spatial Features:** What does the hand look like in a single frame?
2. **Temporal Features:** How does the hand position and shape change over time?

---

## Model Architecture
The architecture consists of a custom **Convolutional Neural Network (CNN)** for feature extraction and a **Long Short-Term Memory (LSTM)** network for sequence processing.

### 1. Spatial Module (CNN)
The CNN acts as a feature extractor, compressing each 64x64 frame into a high-level representation.
* **3 Convolutional Layers:** With Batch Normalization and ReLU activation.
* **Max Pooling:** To reduce dimensionality while retaining key features.
* **Feature Vector:** Each frame is mapped to a 256-dimensional embedding.

### 2. Temporal Module (LSTM)
The LSTM processes the sequence of 16 frame embeddings.
* **Sequence Length:** 16 frames.
* **Hidden Size:** 512.
* **Layers:** 2 LSTM layers with 30% dropout to prevent overfitting.

---

## Dataset: LeapGestRecog
The model utilizes the **Hand Gesture Recognition Database** via Kaggle.
* **10 Gesture Classes:** Palm, L-shape, Fist, Fist Moved, Thumb, Index, OK, Palm Moved, C-shape, and Down.
* **Normalization:** Input data is normalized using ImageNet statistics.

---

## Hardware Optimization
The training and evaluation were performed on an **NVIDIA RTX 3060 GPU**. By utilizing **CUDA acceleration**, the training time was significantly reduced.

---

## Installation & Usage
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/shefojr/CNN-LSTM.git](https://github.com/shefojr/CNN-LSTM.git)

## Results
Training Accuracy: 95.00%

Validation Accuracy: 93.33%

Optimizer: Adam (lr = 0.001)

Loss Function: Cross-Entropy Loss
