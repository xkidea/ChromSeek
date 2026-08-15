# ChromSeek

[中文文档](README_zh.md)

ChromSeek contains a pipeline for various omics prediction tasks, spanning from 1D (multi-omics signal prediction), 2D (TAD and Loop prediction), to 3D (Hi-C enhancement and ChIA-PET prediction) based on genomic sequence and structural features.

The main Hi-C enhancement task lives in `hic_enhancement/`. Downstream prediction
tasks are grouped under the `downstreams/` Python package:

```text
downstreams/
├── tad_prediction/
├── loop_prediction/
├── chiapet_prediction/
└── multiomics_prediction/
```

## Installation

The recommended setup uses a dedicated Conda environment. Python 3.13 is used by the
provided `requirements.txt` pins (Python 3.8+ is supported by the source code).

```bash
conda create -n chromseek python=3.13 pip -y
conda activate chromseek
cd /path/to/chromSeek
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements file installs the CUDA 12.4 PyTorch wheel by default. The host must
have a compatible NVIDIA driver and the process must have access to the GPU (for
example, request a GPU in your scheduler or start a container with GPU passthrough).
Verify the installation with:

```bash
python -c "import torch; print(torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"
```

For a CPU-only machine, edit `requirements.txt` and change `torch==2.6.0+cu124` to
`torch==2.6.0` before running the pip install command. To use the notebooks, register
the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name chromseek --display-name "ChromSeek (chromseek)"
```

Alternatively, if you do not use Conda, install the pinned dependencies directly:

```bash
python -m pip install -r requirements.txt
```

## How to Run

### 1. Download Pre-trained Models
Before running the tutorials, please download the corresponding model weight files (`.pth`) from the [GitHub Releases page](https://github.com/zzzlnb/chromSeek/releases) and place all of them in `checkpoints/`:
- `checkpoints/chromSeek_hic_enhancement.pth`
- `checkpoints/chromSeek_tad_prediction.pth`
- `checkpoints/chromSeek_loop_prediction.pth`
- `checkpoints/chromSeek_chiapet_prediction.pth`
- `checkpoints/transfer_multiomics_best.pth`
- `checkpoints/best_model_448k_200bp.pth` (base checkpoint used for transfer learning)

### 2. Run the Tutorials
The project is organized into modular directories. You can run the tutorial scripts using Python directly from the `chromSeek` root directory. For example:
```bash
python hic_enhancement/tutorial.py
```

## Tutorial Paths
- **Hi-C Enhancement**: `hic_enhancement/tutorial.py`
- **TAD Prediction**: `downstreams/tad_prediction/tutorial.py`
- **Loop Prediction**: `downstreams/loop_prediction/tutorial.py`
- **ChIA-PET Prediction**: `downstreams/chiapet_prediction/tutorial.py`
- **Multi-omics Prediction**: `downstreams/multiomics_prediction/tutorial.py`

### 3. Interactive Single-Cell Analysis Workflow (Jupyter App) 🌟
We provide a user-friendly, end-to-end Jupyter Notebook for **single-cell Micro-C / Hi-C resolution enhancement and TAD structure prediction**. In this interactive application, you can:
- Specify an `mcool` sparse single-cell matrix or a cropped `numpy` array.
- Freely define the target chromosome (e.g., `chr1`) and start region coordinates (using 10kb resolution Bin IDs).
- Automatically extract and align local genomic sequence features (includes local cache support for the `hg38` human genome).
- Call the deep learning framework to perform image reconstruction, and precisely identify high-confidence TAD domains using a dynamic percentile-adaptive mechanism (95th percentile) with visualized comparisons.

**Quick Start**: Access the app by opening `chromSeek_enhancement_app.ipynb` in the root directory using VS Code or a Jupyter server, and run the cells sequentially.
