#!/bin/bash
# Run the Flask app with sudo (for Raspberry Pi)

cd "$(dirname "$0")"
sudo python3 app.py
