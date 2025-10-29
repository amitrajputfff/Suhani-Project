# Fix for macOS threading issues - MUST BE FIRST
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
import tempfile
import sys

# Add paths for custom modules
sys.path.append('Lane_detection')
sys.path.append('Traffic_Sign')
sys.path.append('Vehicle_DC_Final')
sys.path.append('Pedestrian_detection')

from tensorflow.keras.models import load_model
from Lane_detection.custom_layers import spatial_attention, weighted_bce
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

st.set_page_config(page_title="🚗 Complete Autonomous Driving Analysis", page_icon="🚗", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(to right, #667eea, #764ba2);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🚗 Complete Autonomous Driving Analysis</h1>
    <p>Lane Detection • Traffic Signs • Vehicle Classification • Pedestrian Detection</p>
</div>
""", unsafe_allow_html=True)

# Model paths
LANE_MODEL = "Lane_detection/lane_detection_final_6.keras"
YOLO_TRAFFIC_WEIGHTS = "Traffic_Sign/yolov3.weights"
YOLO_TRAFFIC_CFG = "Traffic_Sign/yolov3.cfg"
YOLO_TRAFFIC_NAMES = "Traffic_Sign/coco.names"

IMAGE_SIZE = (224, 224)
vehicle_ids = [2, 3, 5, 7]  # car, motorcycle, bus, truck in COCO

vehicle_class_names = [
    'Auto', 'Bus', 'Empty road', 'Motorcycles',
    'Tempo Traveller', 'Tractor', 'Truck', 'cars'
]

# Load all models
@st.cache_resource
def load_all_models():
    with st.spinner("🔄 Loading all AI models... This may take a minute..."):
        models = {}
        
        # 1. Lane Detection Model
        try:
            models['lane'] = load_model(LANE_MODEL, custom_objects={
                'weighted_bce': weighted_bce,
                'spatial_attention': spatial_attention
            })
            st.success("✅ Lane Detection model loaded")
        except Exception as e:
            st.warning(f"⚠️ Lane Detection model not found: {e}")
            models['lane'] = None
        
        # 2. Traffic Sign Detection (YOLOv3)
        try:
            if os.path.exists(YOLO_TRAFFIC_WEIGHTS):
                models['traffic_net'] = cv2.dnn.readNetFromDarknet(YOLO_TRAFFIC_CFG, YOLO_TRAFFIC_WEIGHTS)
                models['traffic_labels'] = open(YOLO_TRAFFIC_NAMES).read().strip().split("\n")
                ln = models['traffic_net'].getLayerNames()
                models['traffic_ln'] = [ln[i - 1] for i in models['traffic_net'].getUnconnectedOutLayers()]
                st.success("✅ Traffic Sign model loaded")
            else:
                models['traffic_net'] = None
                st.warning("⚠️ Traffic Sign model not found - run traffic_sign_streamlit.py first to download")
        except Exception as e:
            st.warning(f"⚠️ Traffic Sign model error: {e}")
            models['traffic_net'] = None
        
        # 3. Pedestrian Detection (YOLOv8)
        try:
            models['pedestrian'] = YOLO('yolov8n.pt')
            st.success("✅ Pedestrian Detection model loaded")
        except Exception as e:
            st.warning(f"⚠️ Pedestrian Detection model error: {e}")
            models['pedestrian'] = None
        
        # 4. Vehicle Detection (YOLOv8)
        try:
            models['vehicle_yolo'] = YOLO('yolov8n.pt')
            st.success("✅ Vehicle Detection model loaded")
        except Exception as e:
            st.warning(f"⚠️ Vehicle Detection model error: {e}")
            models['vehicle_yolo'] = None
        
        # Vehicle Classification (EfficientNet) - Optional
        try:
            from huggingface_hub import snapshot_download
            model_path = snapshot_download(repo_id="Coder-M/Vehicle")
            models['vehicle_classifier'] = tf.keras.layers.TFSMLayer(model_path, call_endpoint="serving_default")
            st.success("✅ Vehicle Classification model loaded")
        except Exception as e:
            st.info("ℹ️ Vehicle Classification model not available - using YOLO only")
            models['vehicle_classifier'] = None
        
    return models

def process_lane_detection(frame, model):
    """Detect lanes in frame with improved visibility"""
    if model is None:
        return frame
    
    original = frame.copy()
    h, w = frame.shape[:2]
    
    # Preprocess for model
    input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_frame = cv2.resize(input_frame, (640, 360)) / 255.0
    input_frame = np.expand_dims(input_frame, axis=0)
    
    # Get prediction
    result = model.predict(input_frame, verbose=0)
    result = np.squeeze(result)
    result = cv2.resize(result, (w, h))
    
    # IMPROVED: Configurable threshold for better detection
    # Default 0.3, but can be adjusted via settings
    result = (result > 0.3).astype(np.uint8) * 255
    
    # Morphological operations - REDUCED for better detection
    erode_kernel = np.ones((2, 2), np.uint8)
    result = cv2.erode(result, erode_kernel, iterations=1)
    
    # IMPROVED: Smaller kernel for better lane preservation
    open_kernel = np.ones((5, 5), np.uint8)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, open_kernel)
    
    # IMPROVED: Fewer blur iterations
    for _ in range(5):
        result = cv2.GaussianBlur(result, (7, 7), 0)
        _, result = cv2.threshold(result, 100, 255, cv2.THRESH_BINARY)
    
    # IMPROVED: Lower min area for better detection
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(result, connectivity=8)
    result_clean = np.zeros_like(result)
    min_area = 500  # Reduced from 1500
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            result_clean[labels == label] = 255
    
    # IMPROVED: Brighter green overlay with transparency
    result_bgr = cv2.cvtColor(result_clean, cv2.COLOR_GRAY2BGR)
    result_bgr[:, :, 0] = 0  # No blue
    result_bgr[:, :, 1] = result_clean  # Full green where lanes detected
    result_bgr[:, :, 2] = 0  # No red
    
    # IMPROVED: More visible overlay (increased from 0.7 to 1.0)
    return cv2.addWeighted(original, 1, result_bgr, 1.0, 0)

def get_traffic_light_color(roi):
    """Detect traffic light color"""
    if roi.size == 0:
        return "Unknown"
    
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

def process_traffic_signs(frame, net, ln, labels):
    """Detect traffic signs and lights"""
    if net is None:
        return frame
    
    (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
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
            
            if labels[classID] in ["traffic light", "stop sign"] and confidence > 0.3:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.3)
    
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            
            color = (0, 165, 255)  # Orange for traffic signs
            label = f"{labels[classIDs[i]]}"
            
            if labels[classIDs[i]] == "traffic light":
                roi = frame[max(0, y):min(H, y+h), max(0, x):min(W, x+w)]
                light_color = get_traffic_light_color(roi)
                label += f" ({light_color})"
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame

def process_pedestrians(frame, model):
    """Detect pedestrians"""
    if model is None:
        return frame
    
    results = model(frame, classes=[0], conf=0.25, verbose=False)
    
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        
        # Red box for pedestrians
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"Person {conf:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    return frame

def process_vehicles(frame, yolo_model, classifier_model=None):
    """Detect and classify vehicles"""
    if yolo_model is None:
        return frame
    
    results = yolo_model(frame, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy().astype(int)
    confidences = results[0].boxes.conf.cpu().numpy()
    
    for box, cls_id, conf in zip(boxes, classes, confidences):
        if cls_id not in vehicle_ids:
            continue
        
        x1, y1, x2, y2 = map(int, box[:4])
        crop = frame[y1:y2, x1:x2]
        
        if crop.size == 0:
            continue
        
        # Try to classify
        label = "Vehicle"
        if classifier_model is not None:
            try:
                crop_resized = cv2.resize(crop, IMAGE_SIZE)
                crop_input = preprocess_input(np.expand_dims(crop_resized.astype("float32"), axis=0))
                preds = classifier_model(crop_input, training=False)
                preds = preds["output_0"].numpy()
                label = vehicle_class_names[np.argmax(preds)]
            except:
                label = "Vehicle"
        
        # Blue box for vehicles
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return frame

def process_combined_frame(frame, models, lane_threshold=0.3):
    """Apply all 4 detections to a single frame"""
    # 1. Lane Detection (green overlay)
    frame = process_lane_detection(frame, models.get('lane'))
    
    # 2. Traffic Signs (orange boxes)
    if models.get('traffic_net') is not None:
        frame = process_traffic_signs(frame, models['traffic_net'], 
                                      models['traffic_ln'], models['traffic_labels'])
    
    # 3. Pedestrians (red boxes)
    frame = process_pedestrians(frame, models.get('pedestrian'))
    
    # 4. Vehicles (blue boxes)
    frame = process_vehicles(frame, models.get('vehicle_yolo'), 
                            models.get('vehicle_classifier'))
    
    return frame

def process_video(video_path, models):
    """Process entire video with all detections"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("❌ Error opening video file")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output video
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    frames_processed = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process with all 4 detections
        lane_thresh = models.get('lane_threshold', 0.3)
        processed_frame = process_combined_frame(frame, models, lane_thresh)
        out.write(processed_frame)
        
        frames_processed += 1
        progress = int((frames_processed / frame_count) * 100)
        progress_bar.progress(progress)
        status_text.text(f"🎬 Processing: {progress}% ({frames_processed}/{frame_count} frames)")
    
    cap.release()
    out.release()
    
    progress_bar.empty()
    status_text.empty()
    
    return output_path

# Main UI
st.markdown("### 📹 Upload Your Video")
st.markdown("The video will be processed with **all 4 AI models** simultaneously:")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("🛣️ **Lane Detection**\n(Green overlay)")
with col2:
    st.markdown("🚦 **Traffic Signs**\n(Orange boxes)")
with col3:
    st.markdown("🚶 **Pedestrians**\n(Red boxes)")
with col4:
    st.markdown("🚗 **Vehicles**\n(Blue boxes)")

st.markdown("---")

# Settings
with st.expander("⚙️ Detection Settings"):
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        lane_threshold = st.slider("🛣️ Lane Detection Sensitivity", 0.1, 0.9, 0.3, 0.1,
                                   help="Lower = more sensitive (detects more lanes)")
    with col_s2:
        st.markdown("**Tip:** Lower the threshold if lanes aren't being detected")

st.markdown("---")

# Load models
models = load_all_models()
models['lane_threshold'] = lane_threshold  # Pass threshold to models

# File uploader
uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'mov', 'avi', 'mkv'])

if uploaded_file is not None:
    # Save uploaded video
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_input.write(uploaded_file.read())
    temp_input.flush()
    temp_input.close()
    
    # Display original
    st.subheader("📹 Original Video")
    st.video(temp_input.name)
    
    if st.button("🚀 Start Complete Analysis", type="primary"):
        with st.spinner("🔄 Processing video with all AI models..."):
            output_path = process_video(temp_input.name, models)
        
        if output_path:
            st.success("✅ Processing complete!")
            
            st.subheader("🎯 Analyzed Video (All Detections)")
            st.video(output_path)
            
            # Download button
            with open(output_path, 'rb') as f:
                st.download_button(
                    label="📥 Download Analyzed Video",
                    data=f.read(),
                    file_name="autonomous_driving_analysis.mp4",
                    mime="video/mp4"
                )
            
            # Cleanup
            os.unlink(output_path)
    
    # Cleanup
    os.unlink(temp_input.name)

st.markdown("---")
st.markdown("""
### 🎨 Color Legend:
- 🟢 **Green** = Lane markings
- 🟠 **Orange** = Traffic lights & stop signs
- 🔴 **Red** = Pedestrians
- 🔵 **Blue** = Vehicles (classified by type)

**Note:** Processing time depends on video length. Expect ~1-2 minutes per minute of video.
""")

