# ChromSeek

[English](README.md)

ChromSeek 包含一系列组学预测任务的流水线，涵盖了从 1D（多组学信号预测）、2D（TAD、Loop 预测）到 3D（Hi-C 增强、ChIA-PET 预测）等不同的基因组序列和结构特征预测。

Hi-C 增强主任务保留在 `hic_enhancement/`。下游预测任务统一放在
`downstreams/` Python 包中：

```text
downstreams/
├── tad_prediction/
├── loop_prediction/
├── chiapet_prediction/
└── multiomics_prediction/
```

## 环境依赖安装

推荐使用独立的 Conda 环境。`requirements.txt` 中的固定版本以 Python 3.13
为目标（项目源代码本身支持 Python 3.8 及以上版本）。

```bash
conda create -n chromseek python=3.13 pip -y
conda activate chromseek
cd /path/to/chromSeek
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` 默认安装 CUDA 12.4 版本的 PyTorch。主机需要安装兼容的
NVIDIA 驱动，并且当前进程必须获得 GPU 设备权限（例如通过作业调度器申请
GPU，或启动带 GPU 透传的容器）。安装后可以使用以下命令验证：

```bash
python -c "import torch; print(torch.__version__); print('CUDA build version:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"
```

如果只使用 CPU，请在执行安装前将 `requirements.txt` 中的
`torch==2.6.0+cu124` 改为 `torch==2.6.0`。如需在 Jupyter 中使用该环境，注册
一个专用 kernel：

```bash
python -m ipykernel install --user --name chromseek --display-name "ChromSeek (chromseek)"
```

不使用 Conda 时，也可以直接安装固定版本依赖：

```bash
python -m pip install -r requirements.txt
```

## 运行方式

### 1. 下载预训练模型
在此仓库运行相应任务前，请前往项目的 [GitHub Releases 页面](https://github.com/zzzlnb/chromSeek/releases) 下载对应的模型权重文件（`.pth`），并将全部文件放入 `checkpoints/` 目录：
- `checkpoints/chromSeek_hic_enhancement.pth`
- `checkpoints/chromSeek_tad_prediction.pth`
- `checkpoints/chromSeek_loop_prediction.pth`
- `checkpoints/chromSeek_chiapet_prediction.pth`
- `checkpoints/transfer_multiomics_best.pth`
- `checkpoints/best_model_448k_200bp.pth`（迁移学习使用的基础权重）

### 2. 执行测试样例
各个模块均独立组织。您可以在 `chromSeek` 根目录下，直接使用 Python 运行入门教程脚本进行测试：
```bash
python hic_enhancement/tutorial.py
```

## 各个任务教程的路径
- **Hi-C 增强**: `hic_enhancement/tutorial.py`
- **TAD 预测**: `downstreams/tad_prediction/tutorial.py`
- **Loop 预测**: `downstreams/loop_prediction/tutorial.py`
- **ChIA-PET 预测**: `downstreams/chiapet_prediction/tutorial.py`
- **多组学预测**: `downstreams/multiomics_prediction/tutorial.py`

### 3. 交互式单细胞分析工作流 (Jupyter App) 🌟
项目特别提供了一个用户友好的端到端 Jupyter Notebook 工具，适用于**单细胞级别 Micro-C / Hi-C 的分辨率增强及 TAD 结构域预测**。在这个交互应用中，您可以：
- 随时指定 `mcool` 稀疏单细胞矩阵或截取好的 `numpy` 阵列。
- 支持自由设定需要预测的染色体（例如 `chr1`）和起始区域坐标（以 `10kb` 表示的相对 Bin ID）。
- 自动提取本地存储的序列特征予以对齐计算（包含自动利用本地人类全基因组 `hg38` 缓存）。
- 调用深度框架自动完成重构，并在此基础上利用动态百分位自适应机制（95th percentile）定位高置信度 TAD，进行可视化高亮对比。

**快速上手入口**：使用 VS Code 或者官方 Jupyter 服务打开并运行根目录下的 `chromSeek_enhancement_app.ipynb`。
