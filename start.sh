#!/bin/bash
echo "Installing dependencies..."
pip install -r requirements.txt
pip install --force-reinstall opencv-python-headless
echo "Starting bot..."
python main.py
