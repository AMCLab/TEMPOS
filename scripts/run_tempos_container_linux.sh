#!/bin/bash

# Stop on errors
set -e

# Ensure GUI exists
if [ -z "$DISPLAY" ]; then
    echo "Error: DISPLAY is not set. Run inside WSLg or Linux."
    exit 1
fi

# Create exports folder for persistent data
mkdir -p "$(pwd)/exports"

# Grant X11 permission for current user only
xhost +si:localuser:$(whoami) > /dev/null

# Always clean up permissions on exit
trap 'xhost -si:localuser:$(whoami) > /dev/null' EXIT

echo "Starting Tempos ..."

# Detect if GPU is available in this environment (simple heuristic)
GPU_FLAG=""
if command -v nvidia-smi >/dev/null 2>&1; then
    # If nvidia-smi works here, assume GPU passthrough is configured; enable docker GPU flag.
    GPU_FLAG="--gpus all"
fi

# 3. Run the container
# -it            : interactive terminal (logs + Ctrl+C work properly)
# --rm           : auto-remove container after exit
# --network host : allows localhost/127.0.0.1 access (Uvicorn)
# --shm-size=2gb : prevents PyTorch/AI shared-memory crashes
# DISPLAY + X11  : enables GUI windows
# exports mount  : persistent output files on host
docker run -it --rm \
    ${GPU_FLAG} \
    --network host \
    --shm-size=2gb \
    -e DISPLAY="$DISPLAY" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "$(pwd)/exports:/app/exports" \
    ghcr.io/alex-belozerov/tempos:latest
