# 🚀 Autonomous Driving Analysis - Installation Guide

## ⚡ Quick Install (macOS)

### One-Command Installation:

```bash
./install_macos.sh
```

**If you get "permission denied", use:**
```bash
bash install_macos.sh
```

That's it! The script will:
- ✅ Install Homebrew (if needed)
- ✅ Install Python 3.11
- ✅ Create virtual environment
- ✅ Install all Python packages
- ✅ Download AI models (YOLOv3 for traffic signs)
- ✅ Set up everything automatically

**Time:** ~10-15 minutes  
**Internet:** Required

---

## 📋 What Gets Installed:

| Component | Version | Purpose |
|-----------|---------|---------|
| Python | 3.11.14 | Core runtime |
| Streamlit | 1.51.0 | Web UI framework |
| TensorFlow | 2.16.2 | Lane detection (Apple Silicon optimized) |
| OpenCV | 4.12.0.88 | Image/video processing |
| YOLO | 8.3.222 | Object detection |
| YOLOv3 Models | 236 MB | Traffic sign detection |
| HuggingFace Hub | Latest | Model downloads |

**Total Size:** ~2 GB (including models)

---

## 🎯 After Installation:

### Run the Apps:

**Option 1: Interactive Launcher**
```bash
./RUN_APPS.sh
```

**Option 2: Combined Detection (Recommended!)**
```bash
./run_combined.sh
```

**Option 3: Individual Apps**
```bash
./Lane_detection/run_lane_detection.sh
./Traffic_Sign/run_traffic_sign.sh
./Vehicle_DC_Final/run_vehicle_detection.sh
./Pedestrian_detection/run_pedestrian_detection.sh
```

---

## 🛠️ Manual Installation (if needed):

If the automatic script fails, follow these steps:

### 1. Install Homebrew:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. Install Python 3.11:
```bash
brew install python@3.11
```

### 3. Create Virtual Environment:
```bash
python3.11 -m venv venv
source venv/bin/activate
```

### 4. Install Packages:
```bash
pip install --upgrade pip setuptools wheel
pip install streamlit numpy opencv-python
pip install tensorflow-macos tensorflow-metal
pip install ultralytics huggingface-hub matplotlib scikit-learn
```

### 5. Download Traffic Sign Models:
```bash
cd Traffic_Sign
curl -L https://github.com/patrick013/Object-Detection---Yolov3/raw/master/model/yolov3.weights -o yolov3.weights
curl https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg -o yolov3.cfg
curl https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names -o coco.names
cd ..
```

### 6. Make Scripts Executable:
```bash
chmod +x *.sh
chmod +x */run_*.sh
```

---

## ✅ Verify Installation:

```bash
source venv/bin/activate
python --version          # Should be 3.11.x
python -c "import tensorflow as tf; print(tf.__version__)"
python -c "import streamlit; print(streamlit.__version__)"
python -c "from ultralytics import YOLO; print('YOLO OK')"
```

---

## 🐛 Troubleshooting:

### Permission Denied:
```bash
chmod +x install_macos.sh
./install_macos.sh
```

### Python Version Wrong:
```bash
brew install python@3.11
```

### TensorFlow Crashes:
The script installs `tensorflow-macos` which is Apple Silicon optimized.
If on Intel Mac, use:
```bash
pip install tensorflow==2.16.2
```

### Virtual Environment Not Activating:
```bash
rm -rf venv
python3.11 -m venv venv
source venv/bin/activate
```

### Models Not Downloading:
Check internet connection and firewall settings.

---

## 🎓 System Requirements:

- **OS:** macOS 11.0+ (Big Sur or later)
- **RAM:** 8 GB minimum, 16 GB recommended
- **Storage:** 5 GB free space
- **Internet:** Required for installation
- **Processor:** Any (Apple Silicon M1/M2/M3 or Intel)

---

## 📦 What Each App Does:

| App | Input | Output | Time |
|-----|-------|--------|------|
| **Lane Detection** | Video | Green lane overlay | ~1 min/min |
| **Traffic Sign** | Image | Orange boxes with colors | ~2 sec/image |
| **Vehicle Detection** | Image | Blue boxes with types | ~3 sec/image |
| **Pedestrian Detection** | Image/Video | Red boxes | ~5 FPS |
| **COMBINED** ⭐ | Video | All 4 at once! | ~1-2 min/min |

---

## 🎯 First Time Users:

1. Run installation: `./install_macos.sh`
2. Wait for completion (~10-15 min)
3. Read `INSTALLED.txt` for quick start
4. Run: `./run_combined.sh`
5. Upload a video and click "Start Complete Analysis"

---

## 📱 Need Help?

- Check `INSTALLED.txt` after installation
- Read `START_HERE.md` for quick start
- See `QUICK_START.md` for commands
- Full docs in `README_NEW.md`

---

## 🔄 Updating/Reinstalling:

```bash
rm -rf venv
./install_macos.sh
```

This will create a fresh installation with latest packages.

---

**Ready? Run:** `./install_macos.sh`

🚗 Happy Detecting! 🛣️🚦🚶

