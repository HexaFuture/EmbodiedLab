# Third-party software

EmbodiedLab does not redistribute the following software; `./embodiedlab build` and `./embodiedlab install-services`
fetch it from its publishers, and its own licence applies.

| Software | Used for | Source | Licence |
|---|---|---|---|
| NVIDIA Isaac Sim 6.0.1 | base of the lab image (simulation and rendering) | `nvcr.io/nvidia/isaac-sim:6.0.1` | NVIDIA Isaac Sim Additional Software and Materials License |
| SAM3 (commit 2ec3c07) | segmentation service for T1 | https://github.com/facebookresearch/sam3, weights at https://huggingface.co/facebook/sam3 | SAM License |
| cuRobo v0.7.8 | motion planning service for T1 | https://github.com/NVlabs/curobo | NVIDIA Source Code License |
| PyTorch 2.9.1 (CUDA 12.8) | runtime of SAM3 and cuRobo | https://download.pytorch.org | BSD-3-Clause |

Python packages installed from PyPI are listed with their versions in `requirements.txt`, `docker/lab/requirements.txt`
and `services/requirements.txt`; each carries its own licence.

The Franka Emika Panda robot model in `assets/` follows the licence of its upstream description package.
