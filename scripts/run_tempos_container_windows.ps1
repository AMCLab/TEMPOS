# 1. Create a local folder if it doesn't exist
$exportPath = Join-Path $PWD "exports"
if (!(Test-Path $exportPath)) {
    New-Item -ItemType Directory -Path $exportPath | Out-Null
}

Write-Host "Starting Tempos App..." -ForegroundColor Cyan

# 2. Run the container
# Note: $env:DISPLAY grabs the Windows display variable.
# We mount the WSLg sockets to let the Linux GUI bridge to Windows.
docker run -it --rm `
    --network host `
    --gpus all `
    --shm-size=2gb `
    -e DISPLAY=$env:DISPLAY `
    -e WAYLAND_DISPLAY=$env:WAYLAND_DISPLAY `
    -e XDG_RUNTIME_DIR=$env:XDG_RUNTIME_DIR `
    -v /tmp/.X11-unix:/tmp/.X11-unix `
    -v /mnt/wslg:/mnt/wslg `
    -v "${PWD}/exports:/app/exports" `
    ghcr.io/alex-belozerov/tempos:latest
