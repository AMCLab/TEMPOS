#!/bin/bash

# Stop on errors
set -e

# Ensure GUI exists
if [ -z "$DISPLAY" ]; then
    echo "Error: DISPLAY is not set. Run inside WSLg or Linux."
    exit 1
fi

# Create exports folder for persistent data.
# This folder is mounted into the container for persistent data.
mkdir -p "$(pwd)/exports"

# Grant X11 permissions for the GUI.
xhost +local:docker > /dev/null

echo "Starting Tempos ..."

# Run the container.
# -it            : interactive terminal (logs + Ctrl+C work properly)
# --rm           : auto-remove container after exit (no leftovers)
# --network host : allows localhost/127.0.0.1 access (Uvicorn)
# --shm-size=2gb : prevents PyTorch/AI shared-memory crashes
# DISPLAY + X11  : enables GUI windows
# exports mount  : persistent output files on host
#####################################
# To use GPUs, add the following flag:
# --gpus all
#####################################
docker run -it --rm \
    --network host \
    --shm-size=2gb \
    -e DISPLAY="$DISPLAY" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "$(pwd)/exports:/app/exports" \
    ghcr.io/AMCLab/tempos:latest

# Clean up X11 permissions after container exits.
xhost -local:docker > /dev/null
