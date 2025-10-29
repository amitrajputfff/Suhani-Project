#!/bin/bash

# Activate Python 3.11 virtual environment
source venv/bin/activate

# Set threading environment variables
export OMP_NUM_THREADS=1
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

# Run Combined Detection App
streamlit run combined_detection.py

