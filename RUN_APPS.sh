#!/bin/bash

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║   🚗 Autonomous Driving Analysis - App Launcher 🚗        ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${GREEN}All 4 applications are ready to run!${NC}\n"

echo "Select an application to launch:"
echo ""
echo "  1) 🛣️  Lane Detection (Attention CNN)"
echo "  2) 🚦 Traffic Sign Detection (YOLOv3)"
echo "  3) 🚗 Vehicle Detection & Classification (YOLO + EfficientNet)"
echo "  4) 🚶 Pedestrian Detection (YOLOv8)"
echo "  5) 🎯 COMBINED - All 4 in One Video!"
echo "  6) 📊 Master Dashboard (All Apps)"
echo "  7) ❌ Exit"
echo ""

read -p "Enter your choice (1-7): " choice

case $choice in
    1)
        echo -e "\n${YELLOW}Launching Lane Detection...${NC}\n"
        cd Lane_detection && ./run_lane_detection.sh
        ;;
    2)
        echo -e "\n${YELLOW}Launching Traffic Sign Detection...${NC}\n"
        cd Traffic_Sign && ./run_traffic_sign.sh
        ;;
    3)
        echo -e "\n${YELLOW}Launching Vehicle Detection & Classification...${NC}\n"
        cd Vehicle_DC_Final && ./run_vehicle_detection.sh
        ;;
    4)
        echo -e "\n${YELLOW}Launching Pedestrian Detection...${NC}\n"
        cd Pedestrian_detection && ./run_pedestrian_detection.sh
        ;;
    5)
        echo -e "\n${YELLOW}Launching COMBINED Detection (All 4 in One!)...${NC}\n"
        ./run_combined.sh
        ;;
    6)
        echo -e "\n${YELLOW}Launching Master Dashboard...${NC}\n"
        streamlit run run_all_apps.py
        ;;
    7)
        echo -e "\n${GREEN}Goodbye!${NC}\n"
        exit 0
        ;;
    *)
        echo -e "\n${RED}Invalid choice. Please run the script again.${NC}\n"
        exit 1
        ;;
esac

