# Fix for macOS threading issues - MUST BE FIRST, before any imports
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import streamlit as st
import numpy as np
import cv2
cv2.setNumThreads(0)
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from ultralytics import YOLO
import tempfile
from huggingface_hub import snapshot_download

st.set_page_config(page_title="Vehicle Detection & Classification", page_icon="🚗", layout="wide")

IMAGE_SIZE = (224, 224)
REPO_ID = "Coder-M/Vehicle"
vehicle_ids = [2, 3, 5, 7]  # car, motorcycle, bus, truck in COCO

class_names = [
    'Auto',
    'Bus',
    'Empty road',
    'Motorcycles',
    'Tempo Traveller',
    'Tractor',
    'Truck',
    'cars'
]

@st.cache_resource
def load_models():
    """Load YOLO and classification models"""
    with st.spinner("Loading models... This may take a moment on first run."):
        # Download classification model from HuggingFace
        try:
            model_path = snapshot_download(repo_id=REPO_ID)
            classification_model = tf.keras.layers.TFSMLayer(model_path, call_endpoint="serving_default")
        except Exception as e:
            st.error(f"Error loading classification model: {e}")
            st.info("Using basic classification instead")
            classification_model = None
        
        # Load YOLO model
        yolo_model = YOLO("yolov8n.pt")
        
    return yolo_model, classification_model

def classify_vehicles(image, yolo_model, classification_model):
    """Detect and classify vehicles in image"""
    orig_img = image.copy()
    
    # Save image temporarily for YOLO
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    cv2.imwrite(temp_file.name, image)
    
    # YOLO detection
    results = yolo_model(temp_file.name)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy().astype(int)
    confidences = results[0].boxes.conf.cpu().numpy()
    
    os.unlink(temp_file.name)
    
    detections = []
    
    for box, cls_id, conf in zip(boxes, classes, confidences):
        if cls_id not in vehicle_ids:
            continue
        
        x1, y1, x2, y2 = map(int, box[:4])
        crop = orig_img[y1:y2, x1:x2]
        
        if crop.size == 0:
            continue
        
        # Classify the vehicle
        if classification_model is not None:
            try:
                crop_resized = cv2.resize(crop, IMAGE_SIZE)
                crop_input = preprocess_input(np.expand_dims(crop_resized.astype("float32"), axis=0))
                
                preds = classification_model(crop_input, training=False)
                preds = preds["output_0"].numpy()
                
                pred_label = class_names[np.argmax(preds)]
                pred_conf = np.max(preds)
            except:
                pred_label = f"Vehicle (YOLO cls {cls_id})"
                pred_conf = conf
        else:
            pred_label = f"Vehicle (YOLO cls {cls_id})"
            pred_conf = conf
        
        # Draw bounding box and label
        cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_text = f"{pred_label} ({pred_conf:.2f})"
        cv2.putText(orig_img, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        detections.append({
            'label': pred_label,
            'confidence': float(pred_conf),
            'bbox': (x1, y1, x2, y2)
        })
    
    return orig_img, detections

# Streamlit UI
st.title("🚗 Vehicle Detection & Classification")
st.markdown("Upload an image to detect and classify vehicles using YOLO + EfficientNet")

# Load models
yolo_model, classification_model = load_models()

if classification_model is not None:
    st.success("✅ Models loaded successfully!")
else:
    st.warning("⚠️ Classification model not available - using basic YOLO detection only")

# Sidebar
st.sidebar.header("Vehicle Classes")
st.sidebar.markdown("The model can classify:")
for i, class_name in enumerate(class_names, 1):
    st.sidebar.write(f"{i}. {class_name}")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Display original image
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
    
    # Detect and classify
    with st.spinner("Detecting and classifying vehicles..."):
        result_image, detections = classify_vehicles(image, yolo_model, classification_model)
    
    with col2:
        st.subheader("Detection Results")
        st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), use_container_width=True)
    
    # Display detections
    if detections:
        st.success(f"🎯 Found {len(detections)} vehicle(s):")
        
        # Create a summary table
        detection_data = []
        for i, det in enumerate(detections, 1):
            detection_data.append({
                '#': i,
                'Vehicle Type': det['label'],
                'Confidence': f"{det['confidence']:.2%}"
            })
        
        st.table(detection_data)
    else:
        st.info("ℹ️ No vehicles detected in this image.")
    
    # Download button
    _, buffer = cv2.imencode('.jpg', result_image)
    st.download_button(
        label="📥 Download Result",
        data=buffer.tobytes(),
        file_name="vehicle_detection_result.jpg",
        mime="image/jpeg"
    )

st.markdown("---")
st.markdown("""
**How it works:**
1. **YOLOv8** detects vehicles in the image
2. **EfficientNet** classifies each detected vehicle into specific types
3. Results are displayed with bounding boxes and labels
""")

