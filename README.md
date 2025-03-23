# HTM_Comparative_Study

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat&logo=python)  ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat&logo=tensorflow)  ![Numenta](https://img.shields.io/badge/Numenta-HTM-brightgreen?style=flat&logo=numenta)

## Overview

This repository presents a comparative study of Hierarchical Temporal Memory (HTM) implementations, evaluating their performance against state-of-the-art neural networks to assess their feasibility across different tasks. The study covers two primary domains:

- **Image Classification (MNIST):** HTM models are compared with a Convolutional Neural Network (CNN) to benchmark their ability to recognize handwritten digits.
- **Time-Series Anomaly Detection (Hot Gym):** HTM models are evaluated alongside a transformer-based model to analyze their effectiveness in temporal sequence learning and anomaly detection.

This dual comparison provides insights into the strengths and limitations of HTM relative to modern deep learning architectures.

## Implementations Evaluated

- **HTM Core:** A fundamental implementation of HTM algorithms maintained by the HTM community, used for temporal sequence learning and anomaly detection.
- **Other Variants:** Additional modifications and custom implementations to benchmark performance against neural networks.

## Key Scripts

### 1. `mnist.py`

- **🖼️ Purpose:** Applies HTM models to the MNIST dataset, testing their ability to recognize handwritten digits.
- **⚙️ Method:**
  - Converts MNIST images into a format suitable for HTM processing.
  - Implements HTM's Sparse Distributed Representations (SDRs) for classification.
  - Compares classification accuracy with a state-of-the-art CNN.
- **📊 Significance:** Provides an assessment of HTM’s capacity for handling visual pattern recognition tasks relative to modern CNN architectures.

### 2. `hotgym.py`

- **📈 Purpose:** Evaluates HTM’s ability to detect anomalies in temporal data using the "Hot Gym" power consumption dataset.
- **⚙️ Method:**
  - Processes time-series data using HTM’s Temporal Memory model.
  - Learns patterns of power consumption and predicts future values.
  - Compares anomaly detection performance against a transformer-based model.
- **🔍 Significance:** Demonstrates HTM’s capability for real-world anomaly detection and sequence prediction in contrast with advanced transformer models.

## How to Use

### Setting Up the Environment

Ensure you have Python 3.8+ installed. Create and activate a virtual environment:

```bash
python -m venv htm_env
source htm_env/bin/activate  # On Windows use `htm_env\Scripts\activate`
```

Then, install the necessary dependencies:

```bash
pip install -r requirements.txt
```

### Setting Up HTM Core

To compile and install the HTM library from scratch, follow these steps:

```bash
# Clone and install HTM Core
git clone https://github.com/htm-community/htm.core.git
cd htm.core
python htm_install.py
```

### Running the MNIST Experiment

To test HTM on the MNIST dataset, run:

```bash
python mnist.py
```

This script will process the MNIST dataset, apply HTM models, and output classification results. It also compares the performance against a CNN network.

### Running the Hot Gym Experiment

To analyze temporal data and detect anomalies using the "Hot Gym" dataset, run:

```bash
python hotgym.py
```

This script will train an HTM model on the power consumption data, provide anomaly detection insights, and compare its performance against a transformer-based model.

## Results

- **🖼️ MNIST:** Performance benchmarks reveal the differences in handling high-dimensional image data between HTM and conventional CNNs.
- **📈 Hot Gym:** HTM's performance in detecting anomalies and forecasting temporal sequences is contrasted with a transformer-based model, highlighting its strengths and potential limitations.

## References & Acknowledgments

This research was inspired by and built upon the following works:

- [HTM Core](https://github.com/htm-community/htm.core) - The primary HTM implementation used in this study.
- [Biological and Machine Intelligence (BAMI)](https://www.numenta.com/resources/htm/biological-and-machine-intelligence/) - Foundational work on HTM theory.
- [HTM White Paper](https://www.numenta.com/resources/research-publications/papers/hierarchical-temporal-memory-white-paper/) - Detailed explanation of HTM principles.
- [Sparse Distributed Representations (SDRs)](https://arxiv.org/abs/1503.07469) - Core mathematical properties of SDRs in HTM.
- [Prediction-Assisted Cortical Learning Algorithm](https://arxiv.org/abs/1509.08255v2) - Advanced insights into HTM’s computational properties.

