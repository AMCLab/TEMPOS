FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Set essential NVIDIA and Qt variables
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV QT_X11_NO_MITSHM=1

# System Libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3-dev git build-essential ninja-build \
    libgl1-mesa-glx libglib2.0-0 libnss3 libasound2 \
    libx11-6 libxext6 libxrender1 libsm6 libice6 \
    libxcb1 libxcb-xinerama0 libxkbcommon-x11-0 \
    libdbus-1-3 libfontconfig1 libxcb-icccm4 libxcb-image0 \
    libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 \
    libxcb-shape0 libxcb-xfixes0 libxcb-cursor0 \
    libfftw3-dev \
    libegl1 libopengl0 libgl1-mesa-dev \
    tesseract-ocr libtesseract-dev \
    libxrender-dev libx11-xcb-dev libxi-dev \
    libxkbcommon-x11-dev libglu1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python
RUN python -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Create directory for exports and make it world-writable.
RUN mkdir -p /app/exports && chmod 0777 /app/exports

# Copy requirements
COPY requirements.txt .

# Install foundation dependencies (numpy, numba, cython, ninja)
RUN python -m pip install --no-cache-dir \
    $(grep -E '^numpy|^numba' requirements.txt) \
    cython ninja

# Pull out the torch versions from requirements.txt (specific to CUDA 11.8) and install them
RUN python -m pip install --no-cache-dir \
    $(grep -E '^torch|^torchvision|^torchaudio' requirements.txt) \
    --index-url https://download.pytorch.org/whl/cu118

# Install Detectron2
RUN python -m pip install --no-cache-dir --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'

# Install remaining dependancies
RUN python -m pip install --no-cache-dir --no-build-isolation -r requirements.txt

# Copy source code
COPY . /app

# Expose port for Uvicorn
EXPOSE 8001

# Run as default (you will override UID/GID at runtime)
CMD ["python", "main_window.py"]
