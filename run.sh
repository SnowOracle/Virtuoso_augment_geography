#!/bin/bash

# Virtuoso Travel Recommendation System Launcher
echo "Starting Virtuoso Travel Recommendation System..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Check if virtual environment exists, create if not
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create directories if they don't exist
mkdir -p models data api_data

# Run the application
echo "Launching Streamlit application..."
streamlit run app.py

# Deactivate virtual environment on exit
deactivate 