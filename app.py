import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import os
from transformers import TFViTForImageClassification

st.set_page_config(page_title="Satellite AI Analyst", layout="wide")

os.environ['TF_USE_LEGACY_KERAS'] = '1'

MODEL_PATH = "model" 

LABELS_MAP = {
    0: 'AnnualCrop', 1: 'Forest', 2: 'HerbaceousVegetation',
    3: 'Highway', 4: 'Industrial', 5: 'Pasture',
    6: 'PermanentCrop', 7: 'Residential', 8: 'River', 9: 'SeaLake'
}

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at {MODEL_PATH}. Please check the path!")
        return None
    
    try:
        model = TFViTForImageClassification.from_pretrained(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# ==========================================
# 2. INFERENCE FUNCTIONS
# ==========================================
def preprocess_image(image_array):
    """Resizes and normalizes image for ViT (224x224)."""
    img = cv2.resize(image_array, (224, 224))
    img = (img / 127.5) - 1.0
    img = np.transpose(img, (2, 0, 1)) # Channel First
    return np.expand_dims(img, axis=0)

def scan_large_image(image_array, window_size=250, stride=250, threshold=0.85):
    """Runs the Sliding Window 'Zoomed' detection."""
    img_display = image_array.copy()
    h, w, _ = image_array.shape
    
    # Create a progress bar
    progress_bar = st.progress(0)
    total_steps = ((h - window_size) // stride + 1) * ((w - window_size) // stride + 1)
    current_step = 0

    for y in range(0, h - window_size + 1, stride):
        for x in range(0, w - window_size + 1, stride):
            
            # Update Progress
            current_step += 1
            progress_bar.progress(min(current_step / total_steps, 1.0))

            # 1. Cut Patch
            patch = image_array[y:y+window_size, x:x+window_size]
            
            # 2. Predict
            input_tensor = preprocess_image(patch)
            outputs = model.predict({"pixel_values": input_tensor}, verbose=0)
            probs = tf.nn.softmax(outputs.logits, axis=-1)
            
            confidence = np.max(probs)
            predicted_class = int(np.argmax(probs))
            label_text = LABELS_MAP[predicted_class]

            # 3. Draw if confident
            if confidence > threshold:
                # Color generation based on class ID (Consistent colors)
                np.random.seed(predicted_class)
                color = tuple(np.random.randint(0, 255, 3).tolist())
                
                # Draw Box
                cv2.rectangle(img_display, (x, y), (x+window_size, y+window_size), color, 4)
                
                # Draw Label Background & Text
                (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                cv2.rectangle(img_display, (x, y - text_h - 10), (x + text_w, y), color, -1)
                cv2.putText(img_display, label_text, (x, y - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    progress_bar.empty() # Remove bar when done
    return img_display

# ==========================================
# 3. STREAMLIT UI LAYOUT
# ==========================================
st.title("🛰️ Satellite Image AI Analyst")
st.markdown("Upload a satellite image to detect Forests, Highways, Rivers, and more using **Vision Transformers**.")

uploaded_file = st.file_uploader("Choose a satellite image...", type=["jpg", "png", "jpeg", "tif"])

if uploaded_file is not None:
    # Convert file to opencv image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert BGR to RGB

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Analysis Controls")
        
        # MODE 1: SIMPLE CLASSIFICATION
        if st.button("🔍 Quick Classify (Whole Image)"):
            with st.spinner("Analyzing..."):
                input_tensor = preprocess_image(image)
                outputs = model.predict({"pixel_values": input_tensor})
                probs = tf.nn.softmax(outputs.logits, axis=-1)
                top_class = int(np.argmax(probs))
                confidence = np.max(probs)
                
                st.success(f"**Prediction:** {LABELS_MAP[top_class]}")
                st.info(f"**Confidence:** {confidence:.2%}")
                
                # Show bar chart of probabilities
                st.bar_chart({LABELS_MAP[i]: float(probs[0][i]) for i in range(10)})

        # MODE 2: DEEP SCAN (SLIDING WINDOW)
        st.markdown("---")
        st.write("**Deep Feature Scan**")
        scan_size = st.slider("Scan Window Size (Zoom)", min_value=100, max_value=500, value=250, step=50)
        scan_conf = st.slider("Confidence Threshold", min_value=0.5, max_value=1.0, value=0.85)
        
        if st.button("🚀 Run Deep Scan"):
            with st.spinner("Scanning image for features..."):
                result_img = scan_large_image(image, window_size=scan_size, stride=scan_size, threshold=scan_conf)
                st.image(result_img, caption="Detected Features", use_container_width=True)

else:
    st.info("Please upload an image to begin.")