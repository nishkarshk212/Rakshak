#!/bin/bash
echo "Installing dependencies..."
pip install -r requirements.txt
pip install opencv-python-headless
echo "Starting bot..."
python bot.py
