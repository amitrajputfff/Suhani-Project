# 🚗 Autonomous Driving Analysis

Complete AI-powered video analysis for autonomous driving with 4 detection models + combined analysis.

## 🚀 Quick Start

### Install Everything (One Command):
```bash
./install_macos.sh
```

**If you get "permission denied", use:**
```bash
bash install_macos.sh
```

**Time:** 10-15 minutes | **Size:** ~2 GB

### Run the Apps:
```bash
./RUN_APPS.sh          # Interactive launcher
./run_combined.sh      # Combined detection (ALL 4 at once!) ⭐
```

---

## 🎯 What It Does

### 5 Applications:

| # | App | Detection | Color |
|---|-----|-----------|-------|
| 1 | **Lane Detection** | Road lane markings | 🟢 Green |
| 2 | **Traffic Signs** | Lights & stop signs | 🟠 Orange |
| 3 | **Vehicle Detection** | Cars, buses, trucks | 🔵 Blue |
| 4 | **Pedestrian Detection** | People on roads | 🔴 Red |
| 5 | **COMBINED** ⭐ | All 4 in one video! | All colors |

---

## 📦 Tech Stack

- **Python 3.11** - Core runtime
- **Streamlit** - Web UI
- **TensorFlow** - Lane detection (Apple Silicon optimized)
- **YOLOv3/v8** - Object detection
- **OpenCV** - Video processing

---

## ⚡ Installation

### Automatic (Recommended):
```bash
./install_macos.sh
```

Installs: Homebrew, Python 3.11, virtual environment, all packages, AI models.

### Manual:
See `README_INSTALL.md` for detailed instructions.

---

## 🎮 Usage

### Option 1: Combined Analysis (Best!)
```bash
./run_combined.sh
```
Upload a video → All 4 detections in one output!

### Option 2: Individual Apps
```bash
./Lane_detection/run_lane_detection.sh
./Traffic_Sign/run_traffic_sign.sh
./Vehicle_DC_Final/run_vehicle_detection.sh
./Pedestrian_detection/run_pedestrian_detection.sh
```

### Option 3: Interactive Launcher
```bash
./RUN_APPS.sh
```
Choose from menu (1-7).

---

## 📊 Features

✅ **Plug & Play** - One command installation  
✅ **4 AI Models** - Lane, traffic, vehicle, pedestrian  
✅ **Combined Mode** - All detections at once  
✅ **Real-time Progress** - See processing status  
✅ **Adjustable Settings** - Tune detection sensitivity  
✅ **Download Results** - Save processed videos  
✅ **49 Sample Videos** - Ready to test  

---

## 🖥️ System Requirements

- **OS:** macOS 11.0+ (Big Sur or later)
- **RAM:** 8 GB minimum, 16 GB recommended
- **Storage:** 5 GB free space
- **Processor:** Apple Silicon (M1/M2/M3) or Intel

---

## 📁 Project Structure

```
Autonomous-Driving-Analysis-main/
├── install_macos.sh              ← Run this first
├── RUN_APPS.sh                   ← Interactive launcher
├── run_combined.sh               ← Combined detection ⭐
├── combined_detection.py         ← Unified app
├── Lane_detection/               ← App 1
├── Traffic_Sign/                 ← App 2
├── Vehicle_DC_Final/             ← App 3
├── Pedestrian_detection/         ← App 4
├── venv/                         ← Virtual environment
└── README_INSTALL.md             ← Detailed installation
```

---

## 🎯 Quick Example

```bash
# 1. Install (first time only)
./install_macos.sh

# 2. Run combined detection
./run_combined.sh

# 3. Upload video in browser

# 4. Click "Start Complete Analysis"

# 5. Download result with all detections!
```

---

## 🐛 Troubleshooting

### Permission Denied:
```bash
chmod +x install_macos.sh
chmod +x *.sh
chmod +x */run_*.sh
```

### Python Version Wrong:
```bash
source venv/bin/activate
python --version  # Should be 3.11.x
```

### Need to Reinstall:
```bash
rm -rf venv
./install_macos.sh
```

---

## 📚 Documentation

- **README_INSTALL.md** - Installation guide
- **INSTALLED.txt** - Created after installation (quick reference)

---

## 🎓 How It Works

1. **Lane Detection** - Attention-based CNN detects road lanes
2. **Traffic Signs** - YOLOv3 finds lights & signs, HSV detects colors
3. **Vehicle Detection** - YOLOv8 detects vehicles, classifies types
4. **Pedestrian Detection** - YOLOv8 finds people in COCO class 0
5. **Combined** - All 4 models process each frame simultaneously

---

## ⏱️ Performance

- **Lane Detection:** ~0.5 FPS (720p video)
- **Traffic Signs:** ~2 sec per image
- **Vehicle Detection:** ~3 sec per image  
- **Pedestrian Detection:** ~5-10 FPS
- **Combined:** ~1-2 minutes per minute of video

---

## 🎨 Output Colors

- **🟢 Green** = Lane markings (overlay)
- **🟠 Orange** = Traffic lights & stop signs
- **🔴 Red** = Pedestrians
- **🔵 Blue** = Vehicles (with type labels)

---

## 🚀 Ready to Start?

```bash
./install_macos.sh
```

Then run:
```bash
./run_combined.sh
```

**Upload your driving video and see all 4 AI models analyze it!**

---

## 📄 License

See original project license.

---

**Built with TensorFlow • YOLO • OpenCV • Streamlit**

*Optimized for Apple Silicon (M1/M2/M3) with Metal acceleration*

🚗 Happy Detecting! 🛣️🚦🚶

