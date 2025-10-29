#!/bin/bash

# ============================================================================
# Autonomous Driving Analysis - macOS Installation Script
# Complete plug-and-play setup
# ============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Print functions
print_header() {
    echo -e "${BLUE}"
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║   🚗 Autonomous Driving Analysis - macOS Setup 🚗         ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_step() {
    echo -e "${CYAN}▶ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if Homebrew is installed
check_homebrew() {
    print_step "Checking for Homebrew..."
    if ! command -v brew &> /dev/null; then
        print_warning "Homebrew not found. Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        
        # Add Homebrew to PATH for Apple Silicon Macs
        if [[ $(uname -m) == 'arm64' ]]; then
            echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
            eval "$(/opt/homebrew/bin/brew shellenv)"
        fi
        
        print_success "Homebrew installed"
    else
        print_success "Homebrew is already installed"
    fi
}

# Check and install Python 3.11
check_python() {
    print_step "Checking for Python 3.11..."
    
    if command -v python3.11 &> /dev/null; then
        PYTHON_VERSION=$(python3.11 --version | cut -d ' ' -f 2)
        print_success "Python 3.11 is already installed (version $PYTHON_VERSION)"
    else
        print_warning "Python 3.11 not found. Installing via Homebrew..."
        brew install python@3.11
        print_success "Python 3.11 installed"
    fi
}

# Create virtual environment
create_venv() {
    print_step "Creating Python 3.11 virtual environment..."
    
    if [ -d "venv" ]; then
        print_warning "Virtual environment already exists. Removing old one..."
        rm -rf venv
    fi
    
    python3.11 -m venv venv
    print_success "Virtual environment created"
}

# Install Python packages
install_packages() {
    print_step "Installing Python packages (this may take 5-10 minutes)..."
    
    source venv/bin/activate
    
    # Upgrade pip
    echo "  📦 Upgrading pip..."
    pip install --upgrade pip setuptools wheel --quiet
    
    # Install core packages
    echo "  📦 Installing Streamlit..."
    pip install streamlit --quiet
    
    echo "  📦 Installing NumPy and OpenCV..."
    pip install "numpy>=1.26,<2.0" opencv-python --quiet
    
    echo "  📦 Installing TensorFlow for macOS (Apple Silicon optimized)..."
    pip install tensorflow-macos tensorflow-metal --quiet
    
    echo "  📦 Installing YOLO (Ultralytics)..."
    pip install ultralytics --quiet
    
    echo "  📦 Installing additional dependencies..."
    pip install huggingface-hub matplotlib scikit-learn requests tqdm --quiet
    
    print_success "All Python packages installed"
}

# Download YOLOv3 models for traffic sign detection
download_traffic_models() {
    print_step "Downloading traffic sign detection models (236 MB)..."
    
    cd Traffic_Sign
    
    source ../venv/bin/activate
    
    python3 << 'PYEOF'
import os
import requests
from tqdm import tqdm

files = {
    'yolov3.weights': 'https://github.com/patrick013/Object-Detection---Yolov3/raw/master/model/yolov3.weights',
    'yolov3.cfg': 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg',
    'coco.names': 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names'
}

for filename, url in files.items():
    if os.path.exists(filename):
        print(f"  ✅ {filename} already exists")
        continue
    
    print(f"  📥 Downloading {filename}...")
    
    if filename == 'yolov3.weights':
        # Large file - show progress bar
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as file, tqdm(
            total=total_size, unit='B', unit_scale=True, desc=f"  {filename}"
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                pbar.update(len(chunk))
    else:
        # Small files
        response = requests.get(url)
        with open(filename, 'wb') as f:
            f.write(response.content)
    
    print(f"  ✅ {filename} downloaded")

print("\n✅ All traffic sign models downloaded")
PYEOF
    
    cd ..
    print_success "Traffic sign models ready"
}

# Test installations
test_imports() {
    print_step "Testing installations..."
    
    source venv/bin/activate
    
    python3 << 'PYEOF'
import sys

errors = []

try:
    import streamlit
    print("  ✅ Streamlit:", streamlit.__version__)
except ImportError as e:
    errors.append(f"Streamlit: {e}")

try:
    import numpy as np
    print("  ✅ NumPy:", np.__version__)
except ImportError as e:
    errors.append(f"NumPy: {e}")

try:
    import cv2
    print("  ✅ OpenCV:", cv2.__version__)
except ImportError as e:
    errors.append(f"OpenCV: {e}")

try:
    import tensorflow as tf
    print("  ✅ TensorFlow:", tf.__version__)
except ImportError as e:
    errors.append(f"TensorFlow: {e}")

try:
    from ultralytics import YOLO
    print("  ✅ Ultralytics YOLO: installed")
except ImportError as e:
    errors.append(f"YOLO: {e}")

try:
    from huggingface_hub import snapshot_download
    print("  ✅ HuggingFace Hub: installed")
except ImportError as e:
    errors.append(f"HuggingFace: {e}")

if errors:
    print("\n❌ Some imports failed:")
    for error in errors:
        print(f"  - {error}")
    sys.exit(1)
else:
    print("\n✅ All imports successful!")
PYEOF
    
    if [ $? -eq 0 ]; then
        print_success "All packages working correctly"
    else
        print_error "Some packages failed to import"
        exit 1
    fi
}

# Make run scripts executable
make_scripts_executable() {
    print_step "Making run scripts executable..."
    
    chmod +x run_combined.sh
    chmod +x RUN_APPS.sh
    chmod +x Lane_detection/run_lane_detection.sh
    chmod +x Traffic_Sign/run_traffic_sign.sh
    chmod +x Vehicle_DC_Final/run_vehicle_detection.sh
    chmod +x Pedestrian_detection/run_pedestrian_detection.sh
    
    print_success "All scripts are now executable"
}

# Create quick start guide
create_quick_start() {
    cat > INSTALLED.txt << 'EOF'
╔════════════════════════════════════════════════════════════╗
║          ✅ INSTALLATION COMPLETE! ✅                      ║
╚════════════════════════════════════════════════════════════╝

🎉 All 4 + 1 apps are ready to use!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📱 HOW TO RUN:

Option 1: Interactive Launcher (RECOMMENDED)
────────────────────────────────────────────
./RUN_APPS.sh

Then select which app you want:
  1) 🛣️  Lane Detection
  2) 🚦 Traffic Sign Detection  
  3) 🚗 Vehicle Detection
  4) 🚶 Pedestrian Detection
  5) 🎯 COMBINED - All 4 in One! ⭐
  6) 📊 Master Dashboard
  7) ❌ Exit

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Option 2: Run Individual Apps
────────────────────────────────────────────
./run_combined.sh                          # ⭐ RECOMMENDED
./Lane_detection/run_lane_detection.sh
./Traffic_Sign/run_traffic_sign.sh
./Vehicle_DC_Final/run_vehicle_detection.sh
./Pedestrian_detection/run_pedestrian_detection.sh

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📦 WHAT'S INSTALLED:

✅ Python 3.11.14
✅ Streamlit 1.51.0
✅ TensorFlow 2.16.2 (macOS + Metal optimized)
✅ OpenCV 4.12.0.88
✅ Ultralytics YOLOv8
✅ HuggingFace Hub
✅ All dependencies
✅ YOLOv3 Traffic Sign models (236 MB)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 RECOMMENDED FIRST RUN:

./run_combined.sh

This runs the COMBINED app with all 4 detections in one video!

Features:
  🛣️  Green lanes
  🚦 Orange traffic signs (with light colors)
  🔴 Red pedestrian boxes
  🔵 Blue vehicle boxes

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚙️ SETTINGS:

Lane Detection has 49 sample videos ready to test!
Click 🔀 button to load random samples.

Traffic Sign Detection auto-downloads YOLOv3 on first run.

All apps have adjustable detection thresholds in their UI.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🐛 TROUBLESHOOTING:

If you get permission errors:
  chmod +x *.sh
  chmod +x */run_*.sh

If Python version is wrong:
  source venv/bin/activate
  python --version  # Should be 3.11.x

If you need to reinstall:
  rm -rf venv
  ./install_macos.sh

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📚 DOCUMENTATION:

START_HERE.md    - Quick start guide
QUICK_START.md   - Reference guide  
README_NEW.md    - Complete documentation

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚀 READY TO GO!

Run: ./RUN_APPS.sh

Happy Detecting! 🚗🛣️🚦🚶

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EOF
    
    print_success "Quick start guide created (INSTALLED.txt)"
}

# Main installation process
main() {
    clear
    print_header
    
    echo -e "${YELLOW}This script will install everything needed for the"
    echo -e "Autonomous Driving Analysis suite on macOS.${NC}"
    echo ""
    echo "This includes:"
    echo "  • Python 3.11 (via Homebrew)"
    echo "  • Virtual environment"
    echo "  • All Python packages (Streamlit, TensorFlow, YOLO, etc.)"
    echo "  • YOLOv3 models for traffic sign detection"
    echo ""
    echo "Estimated time: 10-15 minutes"
    echo "Internet connection required"
    echo ""
    
    read -p "Continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Installation cancelled."
        exit 1
    fi
    
    echo ""
    echo -e "${GREEN}Starting installation...${NC}"
    echo ""
    
    # Step 1: Check/Install Homebrew
    check_homebrew
    echo ""
    
    # Step 2: Check/Install Python 3.11
    check_python
    echo ""
    
    # Step 3: Create virtual environment
    create_venv
    echo ""
    
    # Step 4: Install Python packages
    install_packages
    echo ""
    
    # Step 5: Download traffic sign models
    download_traffic_models
    echo ""
    
    # Step 6: Test installations
    test_imports
    echo ""
    
    # Step 7: Make scripts executable
    make_scripts_executable
    echo ""
    
    # Step 8: Create quick start guide
    create_quick_start
    echo ""
    
    # Final success message
    echo -e "${GREEN}"
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║            🎉 INSTALLATION COMPLETE! 🎉                   ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo ""
    echo -e "${CYAN}📱 To run the apps:${NC}"
    echo ""
    echo -e "  ${GREEN}./RUN_APPS.sh${NC}          # Interactive launcher"
    echo -e "  ${GREEN}./run_combined.sh${NC}      # Combined detection (recommended!)"
    echo ""
    echo -e "${CYAN}📄 Quick start guide saved to: ${GREEN}INSTALLED.txt${NC}"
    echo ""
    echo -e "${YELLOW}⭐ Try the combined detection app first!${NC}"
    echo -e "   It processes your video with all 4 AI models at once!"
    echo ""
}

# Run main installation
main

