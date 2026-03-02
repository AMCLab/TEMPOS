# Transmission Electron Microscopy Particle Outline Segmentation (TEMPOS)
### Bridging the Gap Between Research and Application

This repostory is for TEMPOS - a GUI for generating synthetic data, appying CTF, training detectron2 models, segmenting experimental images, post processing and exporting data locally with Datasette.

The Video Segmentation directory contains two jupyter notebooks:
* segment_video.ipynb segments a video using a mask-rcnn model, without tracking
* deepsort_video.ipynb segments and tracks a ideo using mask-rcnn model for segmentation and deepSORT for tracking
  
All the paths to the video, model weights and yaml files need to be adjusted to your path.

The ipynb_files directory contains the scripts for using our method without the GUI.

### Start TEMPOS
To run the GUI after setting up the environment:


``` python main_window.py ```

* Create Data page: create your own synthetic data replicating your experimental setup. Don't forget to save each setting.

https://github.com/user-attachments/assets/8c4b5e48-0d5e-4ecf-9a42-f13ffe1932cf

https://github.com/user-attachments/assets/1e481c26-7cba-4544-ae39-eff8ba4a715c
  
* Generate Data: choose from a .json with previous setting or the settings just created. Chose the number of images, and generate your dataset.

https://github.com/user-attachments/assets/3cf01c0c-34f0-4831-a606-14a9b11ed9b2
  
* Apply CTF: optionally apply some CTF to the synthetic images.
* Train Model: Select the path for your training images. The images should be in the same directory as the "instances_train.json" with the annotations.

https://github.com/user-attachments/assets/bddb42e1-b215-4bf3-b9c5-b8191513076d

* Evaluate Model: Select a model and an annotated dataset to evaluate it's performance.
* Segment Images: Select either an individual file or a directory containing .png, .tif, .dm3/dm4 files to segment. Select a train model.

https://github.com/user-attachments/assets/88168ce8-2565-4458-96d3-7dab20db90f3

* Post-Process: Remove scale bar or segment using a sliding window technique to segment smaller particles (pixel-wise). Save the new results.
* Export Datasette: Select a csv file with segmentation results or a .dm3/dm4 directory to expert results and metadata as a searchable sql table, running locally on your web.
  
https://github.com/user-attachments/assets/cf5a1dff-794a-4b98-865e-f22533eb6226

# Setting up the environment

Clone this repo:

```
https://github.com/AMCLab/TEMPOS.git
```

Create a virtual environment with conda and activate the environment:

```
conda create -n tempos python=3.10

conda activate tempos
```
[Install torch](https://pytorch.org/get-started/locally/) according to your nvidia version.

Install Detectron2 (for further info check their [install page](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)):
```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
Install PyQt6 and the extra packages required:
```
pip install -r requirements.txt
```
## Installation on Windows

Installing detectron2 is trickier on windows. Here are the steps that worked on my computer (Windows 11 CUDA Version: 12.3):

```
conda create -n temapp python=3.10

conda activate temapp

conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia

pip install cython

pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI

pip install ninja

conda install -c anaconda cudnn

pip install --upgrade torch

git clone https://github.com/facebookresearch/detectron2.git

python -m pip install -e detectron2

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt

```

# Running as a Docker Container

A containerised version of TEMPOS is available, allowing you to run the GUI without manually installing PyTorch, Detectron2, CUDA, or other dependencies.

**Requirements:**
* Docker (or Docker Desktop on Windows / macOS)
* Display Server:
  - Linux: native X11.
  - Windows: WSLg (standard in Windows 11) or an X-Server like VcXsrv.

### Quick Start

Pull the image from the GitHub Container Registry:
```bash
docker pull ghcr.io/amclab/tempos:latest
```

Run the helper script from the root of the repository:
```bash
bash scripts/run_tempos_container.sh
```

> **IMPORTANT:** To ensure your work isn't lost when the container closes, always save exported files (models, segmentation results, databases) to the `exports/` directory. This folder is automatically mapped to your local machine.
---

### What the Script Does

The script automatically:
1. Verifies `$DISPLAY` is set (required for GUI forwarding)
2. Creates a local `exports/` directory for persistent output
3. Grants Docker temporary access to your X11 server
4. Runs the container with:
   - `--network host` - enables local services such as Uvicorn
   - `--shm-size=2gb` - prevents PyTorch shared memory crashes
   - X11 forwarding and the `exports/` volume mount enabled
5. Revokes X11 permissions once the container exits

---

### GPU Support (Optional)

To enable GPU acceleration, add `--gpus all` to the `docker run` command inside the script.

**Prerequisites:**
- NVIDIA drivers installed on the host
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) configured

If your host lacks a compatible GPU, the container will default to CPU-only mode.

---

### Manual Execution (Advanced)

If you prefer not to use the helper script, you can launch the container manually.

```bash
mkdir -p exports
xhost +local:docker

docker run -it --rm \
    --network host \
    --shm-size=2gb \
    -e DISPLAY="$DISPLAY" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "$(pwd)/exports:/app/exports" \
    ghcr.io/amclab/tempos:latest

xhost -local:docker
```
