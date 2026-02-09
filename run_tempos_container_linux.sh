#!/bin/bash

# 1. Create a local folder if it doesn't exist
mkdir -p "$(pwd)/exports"

# 2. Grant X11 permissions for the GUI
xhost +local:docker > /dev/null

echo "Starting Tempos App..."

# 3. Run the container
# --network host: Solves the 127.0.0.1 Uvicorn issue
# --shm-size: Prevents AI processing crashes
# -v: Handles persistence
docker run -it --rm \
    --network host \
    --shm-size=2gb \
    -e DISPLAY="$DISPLAY" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "$(pwd)/exports:/app/exports" \
    ghcr.io/alex-belozerov/tempos:latest

# 4. Clean up X11 permissions after closing
xhost -local:docker > /dev/null
