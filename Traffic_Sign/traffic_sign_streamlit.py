# Fix for macOS threading issues - MUST BE FIRST, before any imports
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import streamlit as st
import cv2
cv2.setNumThreads(0)
import numpy as np
import urllib.request
import tempfile
from pathlib import Path

st.set_page_config(page_title="Traffic Sign Detection", page_icon="🚦", layout="wide")

# --- Model File Downloader ---
@st.cache_resource
def download_files():
    """Checks for model files and downloads them if they are missing."""
    files = {
        "yolov3.weights": "https://pjreddie.com/media/files/yolov3.weights",
        "yolov3.cfg": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
        "coco.names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
    }
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, (filename, url) in enumerate(files.items(), 1):
        if not os.path.exists(filename):
            status_text.text(f"Downloading {filename}...")
            if filename == "yolov3.weights":
                status_text.text(f"Downloading {filename} (236 MB) - This may take a few minutes...")
            
            try:
                urllib.request.urlretrieve(url, filename)
                status_text.text(f"{filename} downloaded successfully.")
            except Exception as e:
                st.error(f"Error downloading {filename}: {e}")
                st.stop()
        
        progress_bar.progress(idx / len(files))
    
    progress_bar.empty()
    status_text.empty()
    return True

@st.cache_resource
def load_yolo_model():
    """Load YOLO model"""
    download_files()
    
    weights_path = "yolov3.weights"
    config_path = "yolov3.cfg"
    labels_path = "coco.names"
    
    LABELS = open(labels_path).read().strip().split("\n")
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    
    return net, ln, LABELS

def get_traffic_light_color(roi):
    """Detect traffic light color using HSV analysis"""
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    red_lower1 = np.array([0, 120, 70])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 120, 70])
    red_upper2 = np.array([180, 255, 255])
    
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])
    
    green_lower = np.array([40, 70, 70])
    green_upper = np.array([80, 255, 255])
    
    mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
    mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.add(mask_red1, mask_red2)
    
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    
    red_pixels = cv2.countNonZero(red_mask)
    yellow_pixels = cv2.countNonZero(yellow_mask)
    green_pixels = cv2.countNonZero(green_mask)

    if red_pixels > yellow_pixels and red_pixels > green_pixels:
        return "Red"
    elif yellow_pixels > red_pixels and yellow_pixels > green_pixels:
        return "Yellow"
    elif green_pixels > red_pixels and green_pixels > yellow_pixels:
        return "Green"
    
    return "Unknown"

def detect_traffic_signs(image, net, ln, LABELS, confidence_threshold=0.5, nms_threshold=0.3):
    """Detect traffic signs and lights in image"""
    (H, W) = image.shape[:2]
    
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    
    boxes = []
    confidences = []
    classIDs = []
    
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            
            if LABELS[classID] in ["traffic light", "stop sign"] and confidence > confidence_threshold:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
    
    detections = []
    result_image = image.copy()
    
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            
            color = (0, 255, 0)
            label = f"{LABELS[classIDs[i]]}: {confidences[i]:.2f}"
            
            if LABELS[classIDs[i]] == "traffic light":
                roi_y_start, roi_y_end = max(0, y), min(H, y + h)
                roi_x_start, roi_x_end = max(0, x), min(W, x + w)
                if roi_y_end > roi_y_start and roi_x_end > roi_x_start:
                    roi = image[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
                    light_color = get_traffic_light_color(roi)
                    label += f" ({light_color})"
            
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(result_image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            detections.append(label)
    
    return result_image, detections

# Streamlit UI
st.title("🚦 Traffic Sign & Signal Detection")
st.markdown("Upload an image to detect traffic lights and stop signs using YOLOv3")

# Sidebar for parameters
st.sidebar.header("Detection Parameters")
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
nms_threshold = st.sidebar.slider("NMS Threshold", 0.0, 1.0, 0.3, 0.05)

# Load model
with st.spinner("Loading YOLO model..."):
    net, ln, LABELS = load_yolo_model()

st.success("✅ Model loaded successfully!")

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
    
    # Detect
    with st.spinner("Detecting traffic signs and signals..."):
        result_image, detections = detect_traffic_signs(image, net, ln, LABELS, confidence, nms_threshold)
    
    with col2:
        st.subheader("Detected Objects")
        st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), use_container_width=True)
    
    # Display detections
    if detections:
        st.success(f"🎯 Found {len(detections)} object(s):")
        for detection in detections:
            st.write(f"• {detection}")
    else:
        st.info("ℹ️ No traffic lights or stop signs detected in this image.")
    
    # Download button
    _, buffer = cv2.imencode('.jpg', result_image)
    st.download_button(
        label="📥 Download Result",
        data=buffer.tobytes(),
        file_name="traffic_sign_detection_result.jpg",
        mime="image/jpeg"
    )

st.markdown("---")
st.markdown("**Note:** This app detects traffic lights and stop signs using YOLOv3. For traffic lights, it also identifies the color (Red/Yellow/Green).")

