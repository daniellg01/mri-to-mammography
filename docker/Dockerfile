FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ================================
# Qt + X virtual + VNC + noVNC
# ================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libegl1 \
    libopengl0 \
    libglib2.0-0 \
    libx11-6 \
    libxext6 \
    libxi6 \
    libxrender1 \
    libxtst6 \
    libxkbcommon-x11-0 \
    libxkbcommon0 \
    libdbus-1-3 \
    libxcb-cursor0 \
    libxcb-xinerama0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-xfixes0 \
    libfontconfig1 \
    libx11-xcb1 \
    xvfb \
    x11vnc \
    novnc \
    websockify \
    && rm -rf /var/lib/apt/lists/*

# ================================
# App
# ================================
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# ================================
# Patient data (volume)
# ================================
RUN mkdir -p /patient_data
ENV PATIENT_DATA_DIR=/patient_data

# ================================
# Display vars
# ================================
ENV DISPLAY=:0
ENV QT_QPA_PLATFORM=xcb

# noVNC entry point
RUN ln -s /usr/share/novnc/vnc.html /usr/share/novnc/index.html

# Startup script
COPY start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]
