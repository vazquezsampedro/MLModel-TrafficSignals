# Machine Learning Model - Traffic Sign Classification

This project implements a Convolutional Neural Network (CNN) using **PyTorch**, developed and trained within the **Databricks environment**, to classify traffic signs. The system is designed as a foundational module for **autonomous vehicle navigation**, enabling real-time interpretation of traffic signs such as speed limits, stop signs, and other critical road indicators.

---

## 📌 Project Overview

- **Framework:** PyTorch, Databricks
- **Language:** Python
- **Dataset:** [Car Detection Dataset from Kaggle](https://www.kaggle.com/datasets/pkdarabi/cardetection)
- **Purpose:** Predict and classify traffic signs based on labeled image data
- **Author:** Julián Vázquez Sampedro

---

## 🧠 Objective

To build a robust image classification model that assists self-driving systems in identifying road signs. The goal is to develop a scalable ML solution that can be integrated into larger autonomous navigation systems.

---

## 🗂️ Dataset Structure

The dataset includes labeled images organized into:

cardetection/
└── car/
├── train/
│ ├── images/
│ └── labels/
├── valid/
│ ├── images/
│ └── labels/
└── test/
├── images/
└── labels/


Each image is associated with a class ID representing a specific traffic sign.

---

## 🏗️ Model Architecture

A simple CNN architecture was used:

- `Conv2D -> ReLU -> MaxPool`
- `Conv2D -> ReLU -> MaxPool`
- `Flatten -> Linear -> Dropout -> Linear`

Training uses **CrossEntropyLoss** and the **Adam optimizer**.

---

## 🧪 Training & Validation

- **Input size:** 64x64 RGB images
- **Epochs:** 10
- **Batch size:** 32
- **Validation accuracy** tracked after each epoch

Training is executed on **CPU** or **GPU** depending on availability.

---

## 📈 Results & Metrics

- Validation accuracy printed after each epoch
- Distribution of classes displayed before training
- Supports confusion matrix & visual inspection of predictions (optional extension)

---

## 🚀 How to Run

1. Clone this repo and upload it to your Databricks Workspace.
2. Install required libraries:
    ```bash
    pip install torch torchvision kaggle
    ```
3. Add your `kaggle.json` credentials to access the dataset.
4. Download and unzip the dataset to the Databricks working directory.
5. Run the notebook step-by-step or adapt it to your ML pipeline.

---

## 🧩 Future Improvements

- Add data augmentation (rotation, flips)
- Use a pretrained model (e.g., ResNet18) for transfer learning
- Integrate real-time inference with webcam/video input
- Deploy via REST API or on edge devices

---

## 🤝 Contact

julianvazquez171@gmail.com
linkedin.com/in/julianvazquez-sampedro

---
