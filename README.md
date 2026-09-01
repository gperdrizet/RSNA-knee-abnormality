# Kaggle competition development environment

[![Sync release](https://github.com/gperdrizet/kaggle-devcontainer/actions/workflows/sync-release.yml/badge.svg)](https://github.com/gperdrizet/kaggle-devcontainer/actions/workflows/sync-release.yml)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Docker Pulls kaggle-nvidia](https://img.shields.io/docker/pulls/gperdrizet/kaggle-nvidia?label=kaggle-nvidia&logo=docker)](https://hub.docker.com/r/gperdrizet/kaggle-nvidia)
[![Docker Pulls kaggle-cpu](https://img.shields.io/docker/pulls/gperdrizet/kaggle-cpu?label=kaggle-cpu&logo=docker)](https://hub.docker.com/r/gperdrizet/kaggle-cpu)
[![Docker Pulls kaggle-mac](https://img.shields.io/docker/pulls/gperdrizet/kaggle-mac?label=kaggle-mac&logo=docker)](https://hub.docker.com/r/gperdrizet/kaggle-mac)

A ready-to-use Kaggle competition environment for VS Code. Package versions are pinned to match the Kaggle notebook submission environment (Python 3.12), so code developed locally behaves the same when submitted to competitions — no more "works on my machine, fails in the notebook" surprises.

## Requirements

**All users**
- [Docker Desktop](https://docs.docker.com/desktop/) (Windows / Mac) or [Docker Engine](https://docs.docker.com/engine/install/) (Linux)
- [VS Code](https://code.visualstudio.com/) with the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

**NVIDIA GPU users** (also required)
- NVIDIA driver ≥570 ([download](https://www.nvidia.com/Download/index.aspx))
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) *(Linux only, not needed on Windows)*

> **Mac users:** GPU acceleration (Metal/MPS) does not pass through to Docker containers. The Mac configuration uses native ARM64 CPU, no extra setup needed beyond Docker Desktop.

## Quick start

1. **Fork** this repository (click **Fork** at the top of this page)

2. **Clone** your fork:
   ```bash
   git clone https://github.com/<your-username>/kaggle-devcontainer.git
   ```

3. **Open the folder in VS Code**, then open the Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`) and run **Dev Containers: Open Folder in Container**

   > VS Code will ask which configuration to use, pick the one that matches your machine (see table below).

4. **Verify** your setup by running `notebooks/environment_test.ipynb`

## Which configuration should I use?

| If you have... | Choose this |
|----------------|-------------|
| NVIDIA GPU (GTX 10xx / RTX / Quadro / Tesla) | **Kaggle NVIDIA** |
| Windows or Linux machine, no NVIDIA GPU | **Kaggle CPU** |
| Apple Silicon Mac (M1 / M2 / M3 / M4) | **Kaggle Mac** |

Not sure if your GPU is compatible? Check: [NVIDIA CUDA GPUs](https://developer.nvidia.com/cuda-gpus) (need compute capability ≥6.0).

## Kaggle API credentials

The `kaggle` CLI and `kagglehub` are pre-installed. To download competition data and submit from inside the container:

1. Go to [kaggle.com/settings](https://www.kaggle.com/settings) → **API** → **Create New Token** (downloads `kaggle.json`)
2. Place it at `~/.kaggle/kaggle.json` on your host machine and restrict permissions:
   ```bash
   mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
   ```
3. Add a mount to the `.devcontainer/*/devcontainer.json` you use:
   ```json
   "mounts": [
     "source=${localEnv:HOME}/.kaggle,target=/home/vscode/.kaggle,type=bind"
   ]
   ```
4. Rebuild the container, then verify:
   ```bash
   kaggle competitions list
   ```

> Keep `kaggle.json` out of your repository — never commit credentials.

## Using as a template for new projects

Fork this repo once, then use it as a GitHub template to spin up new projects instantly.

### One-time setup

1. Go to your fork on GitHub
2. Click **Settings** → scroll to **Template repository** → enable it

### Creating a new project

1. Go to your fork and click **Use this template** → **Create a new repository**
2. Name your new repo and click **Create repository**
3. Clone it and start working:
   ```bash
   git clone https://github.com/<your-username>/my-new-project.git
   ```

4. **Clean it up** - remove anything that doesn't belong to your project:
   - Update `README.md` to describe your project
   - Delete unused devcontainer configs (e.g. if you only use CPU, remove `nvidia/` and `mac/`)
   - Remove or replace `notebooks/environment_test.ipynb` with your own notebooks
   - Delete test data from `data/`
   ```bash
   git add -A && git commit -m "Initial project setup" && git push
   ```

## Adding Python packages

### Temporary (lost on container rebuild)

```bash
pip install <package-name>
```

### Permanent (recommended)

1. Create a `requirements.txt` in the repository root:
   ```
   umap-learn
   hdbscan
   ```

   > Note: the environment intentionally pins versions to match Kaggle's notebook runtime. If you add or upgrade packages, your local environment may drift from the submission environment.

2. Add a `postCreateCommand` to the relevant `.devcontainer/*/devcontainer.json`:
   ```json
   "postCreateCommand": "pip install -r requirements.txt"
   ```

3. Rebuild the container (`Ctrl+Shift+P` → **Dev Containers: Rebuild Container**)

## Keeping your fork updated

```bash
# Add upstream once
git remote add upstream https://github.com/gperdrizet/kaggle-devcontainer.git

# Pull in updates
git fetch upstream && git merge upstream/main
```

## What's included

All versions pinned to match the Kaggle notebook environment:

| Package | Version | Purpose |
|---------|---------|---------|
| numpy, pandas, polars, scipy | 2.0.2 / 2.3.3 / 1.35.2 / 1.16.3 | Core data stack |
| scikit-learn, statsmodels | 1.6.1 / 0.14.6 | Machine learning and statistics |
| xgboost, lightgbm, catboost | 3.2.0 / 4.6.0 / 1.2.10 | Gradient boosting |
| torch, torchvision, torchaudio | 2.10.0 / 0.25.0 / 2.10.0 | Deep learning (PyTorch) |
| tensorflow, keras | 2.20.0 / 3.13.2 | Deep learning (TensorFlow) |
| transformers, datasets, accelerate, peft | 5.0.0 / 5.0.0 / 1.13.0 / 0.19.1 | HuggingFace ecosystem |
| opencv, albumentations, scikit-image | 4.13.0.92 / 2.0.8 / 0.25.2 | Computer vision |
| matplotlib, seaborn, plotly | 3.10.0 / 0.13.2 / 5.24.1 | Visualization |
| optuna, shap | 4.9.0 / 0.51.0 | Tuning and explainability |
| kaggle, kagglehub | 2.0.2 / 1.0.0 | Kaggle API and dataset access |
| cuDF, cuML, CuPy | 26.2.1 / 26.2.0 / 14.0.1 | RAPIDS GPU acceleration (NVIDIA only) |
| jupyterlab | 3.6.8 | Interactive notebooks (the version Kaggle ships) |

## GPU compatibility (NVIDIA)

Requires compute capability ≥6.0 (Pascal / GTX 10xx or newer):

| Architecture | Example GPUs | Compute Capability | PyTorch | RAPIDS (cuDF/cuML) |
|--------------|--------------|-------------------|---------|--------------------|
| Pascal | GTX 1050–1080, Tesla P100 | 6.0–6.1 | ✅ | ❌ |
| Volta | Tesla V100, Titan V | 7.0 | ✅ | ✅ |
| Turing | RTX 2060–2080, GTX 1660, T4 | 7.5 | ✅ | ✅ |
| Ampere | RTX 3060–3090, A100 | 8.0–8.6 | ✅ | ✅ |
| Ada Lovelace | RTX 4060–4090 | 8.9 | ✅ | ✅ |
| Hopper | H100, H200 | 9.0 | ✅ | ✅ |
| Blackwell | RTX 5070–5090, B100, B200 | 10.0 | ✅ | ✅ |

> **Pascal note:** this environment ships a custom PyTorch 2.10.0 build with Pascal support — torch works on your P100/GTX 10xx here even though it is broken in Kaggle's own P100 notebook sessions (their stock wheel has no Pascal kernels). RAPIDS requires Volta or newer everywhere, including on Kaggle itself.

## Project structure

```
kaggle-devcontainer/
├── .devcontainer/
│   ├── nvidia/
│   │   └── devcontainer.json   # NVIDIA GPU configuration
│   ├── cpu/
│   │   └── devcontainer.json   # CPU configuration
│   └── mac/
│       └── devcontainer.json   # Mac (ARM64) configuration
├── data/                       # Store datasets here
├── notebooks/
│   └── environment_test.ipynb  # Verify your setup
├── .gitignore
├── LICENSE
└── README.md
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Docker won't start | Enable virtualization in BIOS / enable WSL2 on Windows |
| Permission denied (Linux) | Add your user to the docker group, then log out and back in |
| GPU not detected | Update NVIDIA drivers (≥570); Linux: install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) |
| Container build fails | Check your internet connection |
| Module not found | Add the package to `requirements.txt` and rebuild the container |
