#!/bin/bash

# Activate Python 3.11 virtual environment
source ../venv/bin/activate

# Set threading environment variables
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

# Run Streamlit
streamlit run Lane_detection_website.py

