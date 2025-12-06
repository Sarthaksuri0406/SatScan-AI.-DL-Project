# SATSCAN: Satellite Image AI Analyst
SATSCAN is a deep learning application designed for the automated analysis and classification of geospatial imagery. Leveraging a fine-tuned Vision Transformer (ViT) architecture, the system provides high-accuracy land cover classification and implements a sliding-window inference engine to map semantic features across large-scale satellite maps.

## Table of Contents

1.  [Overview](https://www.google.com/search?q=%23overview)
2.  [Key Features](https://www.google.com/search?q=%23key-features)
3.  [Performance](https://www.google.com/search?q=%23performance)
4.  [Installation](https://www.google.com/search?q=%23installation)
5.  [Usage](https://www.google.com/search?q=%23usage)
6.  [Technical Architecture](https://www.google.com/search?q=%23technical-architecture)
7.  [Dataset](https://www.google.com/search?q=%23dataset)
8.  [License](https://www.google.com/search?q=%23license)

## Overview

Automated interpretation of satellite imagery is critical for environmental monitoring, urban planning, and disaster response. SATSCAN automates this process by dividing high-resolution imagery into processable patches, classifying them using a Transformer-based attention mechanism, and reconstructing the semantic map in real-time.

**View Demo:**
*(Place a screenshot of your Streamlit app here. For example: `![Dashboard Screenshot](assets/screenshot.png)`)%*

## Key Features

  * **Vision Transformer Backend:** Deploys `vit-base-patch16-224` pretrained on ImageNet and fine-tuned on geospatial data.
  * **Multi-Class Segmentation:** Capable of distinguishing 10 distinct terrain classes including Industrial, Residential, Forest, and Highway.
  * **Deep Scan Engine:** A custom sliding-window algorithm that processes large-scale images (e.g., 4000x4000px) by analyzing local context windows (250x250px) with configurable stride.
  * **Confidence Visualization:** Filters predictions based on a user-defined probability threshold to reduce false positives.
  * **Legacy Compatibility:** Engineered to bridge the gap between TensorFlow 2.16+ (Keras 3) and Hugging Face Transformers.

## Performance

The model was evaluated on the EuroSAT validation set using the following metrics:

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 98.5% |
| **F1-Score** | 0.98 |
| **Inference Time** | \~150ms per patch (GPU) |

*Note: Performance may vary based on the specific hardware acceleration available.*

## Installation

### 1\. Clone the Repository

```bash
git clone https://github.com/yourusername/SATSCAN.git
cd SATSCAN
```

### 2\. Environment Setup

It is recommended to use a virtual environment to manage dependencies.

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3\. Install Dependencies

```bash
pip install -r requirements.txt
```

*If `requirements.txt` is missing, install manually:*

```bash
pip install tensorflow transformers opencv-python streamlit numpy tf_keras
```

## Usage

### Model Configuration

Ensure the fine-tuned model artifacts are present in the project directory:

  * `eurosat_vit_augmented/config.json`
  * `eurosat_vit_augmented/tf_model.h5`

### Launching the Interface

Execute the Streamlit application:

```bash
streamlit run app.py
```

The interface will be accessible at `http://localhost:8501`.

### Inference Modes

1.  **Quick Classify:** Analyzes the global context of a single uploaded image patch.
2.  **Deep Scan:** Iteratively scans a high-resolution map to detect and label specific features (e.g., locating a highway within a forest).

## Technical Architecture

The solution implements a strict pipeline to ensure data consistency between training and inference:

1.  **Preprocessing:**

      * Resizing: Bicubic interpolation to 224x224.
      * Normalization: Pixel intensity scaling to range [-1, 1].
      * Channel Ordering: Transposition from (H, W, C) to (C, H, W) to satisfy ViT requirements.

2.  **Augmentation Strategy:**
    During training, the model utilized random spatial transformations (rotation, zoom, flip) and photometric distortions (contrast) to improve generalization on unseen satellite data.

## Dataset

This project utilizes the **EuroSAT** dataset.

> Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019). EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing.

## License

Distributed under the MIT License. See `LICENSE` for more information.

-----
**Institution:** Thapar Institute of Engineering and Technology
