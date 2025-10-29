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
import tempfile
from ultralytics import YOLO

st.set_page_config(page_title="Pedestrian Detection", page_icon="🚶", layout="wide")

@st.cache_resource
def load_yolo_model():
    """Load YOLOv8 model for pedestrian detection"""
    model = YOLO('yolov8n.pt')
    return model

def detect_pedestrians_image(image, model, conf_threshold=0.15):
    """Detect pedestrians in a single image"""
    results = model(image, classes=[0], conf=conf_threshold)  # class 0 is 'person'
    
    result_img = image.copy()
    detections = []
    
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])
        
        # Draw red bounding box
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"Person {confidence:.2f}"
        cv2.putText(result_img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        detections.append({
            'confidence': confidence,
            'bbox': (x1, y1, x2, y2)
        })
    
    return result_img, detections

def process_video(video_path, model, conf_threshold=0.15):
    """Process video and detect pedestrians"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("Error opening video file")
        return None, 0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create temporary output file
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    frames_processed = 0
    total_detections = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect pedestrians
        results = model(frame, classes=[0], conf=conf_threshold)
        
        # Draw bounding boxes
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            total_detections += 1
        
        out.write(frame)
        frames_processed += 1
        
        # Update progress
        progress = int((frames_processed / frame_count) * 100)
        progress_bar.progress(progress)
        status_text.text(f"Processing: {progress}% ({frames_processed}/{frame_count} frames)")
    
    cap.release()
    out.release()
    
    progress_bar.empty()
    status_text.empty()
    
    return output_path, total_detections

# Streamlit UI
st.title("🚶 Pedestrian Detection")
st.markdown("Detect pedestrians in images or videos using YOLOv8")

# Load model
with st.spinner("Loading YOLO model..."):
    model = load_yolo_model()

st.success("✅ Model loaded successfully!")

# Sidebar
st.sidebar.header("Detection Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.15, 0.05)

# Choose mode
mode = st.radio("Select Mode:", ["Image Detection", "Video Detection"])

if mode == "Image Detection":
    st.subheader("📷 Image Pedestrian Detection")
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
        
        # Detect pedestrians
        with st.spinner("Detecting pedestrians..."):
            result_image, detections = detect_pedestrians_image(image, model, conf_threshold)
        
        with col2:
            st.subheader("Detection Results")
            st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        # Display results
        if detections:
            st.success(f"🎯 Detected {len(detections)} pedestrian(s)")
            
            # Show details
            with st.expander("Detection Details"):
                for i, det in enumerate(detections, 1):
                    st.write(f"**Pedestrian {i}:** Confidence = {det['confidence']:.2%}")
        else:
            st.info("ℹ️ No pedestrians detected in this image.")
        
        # Download button
        _, buffer = cv2.imencode('.jpg', result_image)
        st.download_button(
            label="📥 Download Result",
            data=buffer.tobytes(),
            file_name="pedestrian_detection_result.jpg",
            mime="image/jpeg"
        )

else:  # Video Detection
    st.subheader("🎥 Video Pedestrian Detection")
    uploaded_video = st.file_uploader("Choose a video...", type=['mp4', 'mov', 'avi', 'mkv'])
    
    if uploaded_video is not None:
        # Save uploaded video to temp file
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_input.write(uploaded_video.read())
        temp_input.flush()
        temp_input.close()
        
        # Display original video
        st.subheader("Original Video")
        st.video(temp_input.name)
        
        # Process button
        if st.button("🚀 Start Detection"):
            with st.spinner("Processing video... This may take a while."):
                output_path, total_detections = process_video(temp_input.name, model, conf_threshold)
            
            if output_path:
                st.success(f"✅ Processing complete! Total pedestrian detections: {total_detections}")
                
                # Display result video
                st.subheader("Processed Video")
                st.video(output_path)
                
                # Download button
                with open(output_path, 'rb') as f:
                    st.download_button(
                        label="📥 Download Processed Video",
                        data=f.read(),
                        file_name="pedestrian_detection_video.mp4",
                        mime="video/mp4"
                    )
                
                # Cleanup
                os.unlink(output_path)
        
        # Cleanup input
        os.unlink(temp_input.name)

st.markdown("---")
st.markdown("""
**How it works:**
- Uses **YOLOv8** to detect people (class 0 in COCO dataset)
- Red bounding boxes indicate detected pedestrians
- Adjustable confidence threshold for detection sensitivity
- Supports both images and videos
""")

