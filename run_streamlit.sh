#!/bin/bash

# Kill any existing Streamlit processes
echo "Stopping any existing Streamlit processes..."
pkill -f "streamlit run" || true

# Path to your virtual environment
VENV_PATH="./venv"  # Update this to your virtual environment path

# Activate the virtual environment
source "$VENV_PATH/bin/activate"

# Run the Streamlit app with a specific port
# Using a high port number to avoid conflicts
echo "Starting Streamlit on port 9876..."
streamlit run app.py --server.port 9876

# Deactivate virtual environment when the app is closed
deactivate 