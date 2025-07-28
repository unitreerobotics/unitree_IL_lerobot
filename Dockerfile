# Base image with CUDA 12.2, cuDNN, and Ubuntu 22.04
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    git \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3.10 -m pip install --upgrade pip

# Set the working directory
WORKDIR /workspace

# Clone unitree_IL_lerobot repository with submodules
RUN git clone --recurse-submodules https://github.com/unitreerobotics/unitree_IL_lerobot.git

# Clone unitree_sdk2_python repository
RUN git clone https://github.com/unitreerobotics/unitree_sdk2_python.git

# Install LeRobot (editable mode)
RUN pip install -e unitree_IL_lerobot/unitree_lerobot/lerobot

# Install unitree_lerobot (editable mode)
RUN pip install -e unitree_IL_lerobot

# Install unitree_sdk2_python (editable mode)
RUN pip install -e unitree_sdk2_python

# Install PyTorch with CUDA 12.1 compatibility
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Set default shell
CMD ["/bin/bash"]

