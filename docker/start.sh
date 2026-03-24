#!/bin/bash
set -e

echo "Starting virtual display..."
Xvfb :0 -screen 0 1920x1080x24 &

echo "Starting VNC server..."
x11vnc -display :0 -nopw -forever -shared &

echo "Starting noVNC..."
websockify --web=/usr/share/novnc 6080 localhost:5900 &

echo "Starting Qt app..."
python main.py
