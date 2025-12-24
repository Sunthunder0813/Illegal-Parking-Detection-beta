#!/bin/bash

set -e

# Check for Raspberry Pi OS / ARM64
ARCH=$(uname -m)
if [[ "$ARCH" != "aarch64" ]]; then
    echo "Warning: This script is intended for Raspberry Pi OS (64-bit, ARM64). Detected architecture: $ARCH"
    echo "Continue at your own risk."
fi

# Ensure locale is set for non-interactive installs
export DEBIAN_FRONTEND=noninteractive
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

echo "Updating system..."
sudo apt update && sudo apt upgrade -y

echo "Installing Python 3, pip, and system dependencies..."
sudo apt install -y python3 python3-pip python3-venv \
    libopencv-dev python3-opencv \
    libatlas-base-dev libhdf5-dev libhdf5-serial-dev \
    libjpeg-dev libpng-dev libtiff-dev libavcodec-dev libavformat-dev libswscale-dev \
    libgtk2.0-dev pkg-config libavresample-dev libxvidcore-dev libx264-dev \
    libtbb2 libtbb-dev libdc1394-22-dev libv4l-dev \
    libopenblas-dev liblapack-dev gfortran \
    ffmpeg wget unzip git

echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing Python dependencies..."
pip install numpy flask opencv-python

echo "Installing additional Python packages for your project..."
pip install requests

# Install Hailo SDK if .deb file is present
if [ -f "hailort_4.23.0_arm64.deb" ]; then
    echo "Found hailort_4.23.0_arm64.deb, installing Hailo SDK..."
    sudo dpkg -i hailort_4.23.0_arm64.deb || sudo apt-get install -f -y
    echo "Hailo SDK installed."
else
    echo "Hailo SDK .deb file not found. Please download hailort_4.23.0_arm64.deb and place it in this directory if you need Hailo support."
fi

echo "Creating violation images directory if not exists..."
mkdir -p static/violations

echo "Setup complete!"
echo "To activate your Python environment, run: source venv/bin/activate"
echo "If this is your first boot, please reboot after setup for all changes to take effect."
