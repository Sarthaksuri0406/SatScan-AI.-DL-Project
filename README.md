# SATSCAN: Satellite Image AI Analyst

SATSCAN is a deep learning application designed to analyze and classify complex satellite imagery. Powered by a fine-tuned Vision Transformer (ViT), it performs land cover classification on the EuroSAT dataset and features a sliding-window inference engine to detect and map features (such as highways, rivers, and forests) across large-scale satellite maps.

## Features

  * **State-of-the-Art Architecture:** Utilizes the `vit-base-patch16-224` Vision Transformer model fine-tuned for geospatial data.
  * **10-Class Classification:** Accurately identifies terrain types including Annual Crop, Forest, Highway, Industrial, Pasture, Residential, River, and Sea/Lake.
  * **Deep Scan Technology:** Implements a sliding window algorithm to process high-resolution satellite imagery, detecting multiple terrain features within a single image context.
  * **Data Augmentation Pipeline:** Trained with random rotations, flips, zooms, and contrast adjustments to ensure robustness against varying satellite capture conditions.
  * **Interactive Web Interface:** Built with Streamlit to provide real-time analysis, confidence visualization, and tunable scanning parameters.

## Tech Stack

  * **Deep Learning:** TensorFlow (Keras), Hugging Face Transformers
  * **Computer Vision:** OpenCV, NumPy
  * **Web Framework:** Streamlit
  * **Model:** Google Vision Transformer (ViT)

## Installation

### Prerequisites

Ensure you have Python 3.8 or higher installed.

### 1\. Clone the Repository

```bash
git clone https://github.com/yourusername/SATSCAN.git
cd SATSCAN
```

### 2\. Set Up Virtual Environment (Recommended)

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3\. Install Dependencies

```bash
pip install tensorflow transformers opencv-python streamlit numpy
```

*Note: This project requires the `tf-keras` legacy package for compatibility with Hugging Face Transformers.*

## Usage

### 1\. Download the Model

Ensure your trained model files (`config.json` and `tf_model.h5`) are placed in the `eurosat_vit_augmented` directory. If you have not trained the model yet, run the training script provided in the notebooks folder.

### 2\. Run the Application

Execute the following command in your terminal to launch the web interface:

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

## Project Structure

```text
SATSCAN/
├── app.py                     # Main Streamlit application entry point
├── eurosat_vit_augmented/     # Directory containing the fine-tuned model
│   ├── config.json
│   └── tf_model.h5
├── requirements.txt           # List of dependencies
└── README.md                  # Project documentation
```

## How It Works

### Training

The model was fine-tuned on the EuroSAT dataset (27,000 labeled satellite images). We replaced the original classification head of the pre-trained ViT model with a custom 10-class layer. The training pipeline utilizes `tf.data` for efficient loading and includes a custom preprocessing step to resize images to 224x224 and normalize pixel values to the range [-1, 1].

### Inference (Deep Scan)

For large satellite images, standard resizing causes loss of detail. SATSCAN solves this by using a sliding window approach:

1.  **Patch Extraction:** The system scans the large image using a configurable window (e.g., 250x250 pixels).
2.  **Processing:** Each patch is resized and fed into the ViT model.
3.  **Filtering:** Predictions are filtered based on a confidence threshold (default \> 85%).
4.  **Visualization:** Bounding boxes and labels are drawn on the original image to highlight detected features.

## Known Issues & Fixes

**Optimizer Compatibility:**
TensorFlow 2.16+ uses Keras 3 by default, which conflicts with some Hugging Face Transformers models. This project enforces Legacy Keras mode. If you encounter optimizer errors, ensure the following environment variable is set at the top of your execution script (already included in `app.py`):

```python
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
```

## Dataset

This project uses the EuroSAT dataset: A novel dataset and deep learning benchmark for land use and land cover classification. Patrick Helber, Benjamin Bischke, Andreas Dengel, Damian Borth. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2019.

## License

This project is licensed under the MIT License.
