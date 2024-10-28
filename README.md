# Deep Learning for Network Performance Analysis

## Project Overview
This project applies deep learning techniques, specifically LSTM (Long Short-Term Memory) networks, to analyze and predict network performance metrics. The goal is to improve network monitoring and anomaly detection using historical traffic data. By leveraging deep learning, the model aims to provide accurate real-time predictions of latency, throughput, and other critical network parameters.

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model Training](#model-training)
- [Results](#results)
- [Contributions](#contributions)
- [License](#license)

## Requirements
To run this project, you need the following Python packages:
- numpy
- pandas
- seaborn
- matplotlib
- scikit-learn
- tensorflow

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/anirudh7371/Deep-Learning-for-Network-Performance-Analysis
   cd Deep-Learning-for-Network-Performance-Analysis
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset
The dataset used in this project can be found at [Kaggle: Network Intrusion Dataset](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset). It consists of multiple CSV files containing network traffic data from various days of the week. The dataset files are:
- `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
- `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`
- `Friday-WorkingHours-Morning.pcap_ISCX.csv`
- `Monday-WorkingHours.pcap_ISCX.csv`
- `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv`
- `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`
- `Tuesday-WorkingHours.pcap_ISCX.csv`
- `Wednesday-workingHours.pcap_ISCX.csv`

Ensure the dataset is in the specified directory before running the code.

## Usage
1. Load the dataset and perform exploratory data analysis (EDA).
2. Clean the data by removing duplicates and handling missing values.
3. Scale the features for better model performance.
4. Train the LSTM model and evaluate its performance using classification metrics.

## Model Training
The model is structured as follows:
- Input Layer: Accepts reshaped feature data.
- LSTM Layers: Two LSTM layers with dropout for regularization.
- Dense Layer: Fully connected layer with ReLU activation.
- Output Layer: Softmax activation for multi-class classification.

The model is trained using categorical cross-entropy as the loss function and Adam optimizer.

## Results
After training, the model's performance is evaluated using:
- Accuracy
- Confusion Matrix
- Precision-Recall Curve
- ROC Curve

These metrics provide insight into the model's ability to predict network performance accurately.

## Contributions
Contributions are welcome! Feel free to fork the repository and submit pull requests.

---
