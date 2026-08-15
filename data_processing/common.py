import gzip
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
from scipy.sparse import coo_matrix


BIN_SIZE = 1000
HIC_RESOLUTION = 10000
NUM_HIC_BINS = 224
DNA_BP_WINDOW = HIC_RESOLUTION * NUM_HIC_BINS
NUM_1KB_BINS = DNA_BP_WINDOW // BIN_SIZE

ALL_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
FULL_SPLITS = {
    "train": [f"chr{i}" for i in range(1, 18)],
    "val": ["chr18", "chr19"],
    "test": ["chr20", "chr21", "chr22", "chrX"],
}
SMOKE_CHROMS = ["chr22"]

DEFAULT_HIC_ROOT = "/mnt/nfs/jyzhu/dataset/lets_fk_hic/hic_patches_10kb_multiratio"
DEFAULT_SEQ_ROOT = "/mnt/nfs/jyzhu/proj/ChromSeek/DNA_ChromSeek/DNA_only_10kb/genome_cache"
DEFAULT_BEDPE_DIR = "/mnt/nfs/jyzhu/dataset/lets_fk_hic/ENCODE_Data"
DEFAULT_MULTIOMICS_HIC_ROOTS = {
    "GM12878": "/mnt/nfs/jyzhu/dataset/lets_fk_hic/hic_patches_10kb_multiratio/GM12878_Hic",
    "K562": "/mnt/nfs/jyzhu/dataset/lets_fk_hic/hic_patches_10kb_multiratio/K562_Hic",
}
DEFAULT_MULTIOMICS_PREPROCESSED_DIR = str(
    Path(__file__).resolve().parents[1] / "downstreams" / "multiomics_prediction" / "preprocessed_multicell"
)

MULTIOMICS_TRACK_NAMES = [
    "ATAC",
    "CTCF",
    "H3K27ac",
    "H3K27me3",
    "H3K36me3",
    "H3K4me1",
    "H3K4me3",
    "H3K79me2",
    "H3K9me3",
    "H4K20me1",
    "MYC",
    "POLR2A",
    "RAD21",
    "SMC3",
    "WGBS",
]


TAD_BEDPE_FILENAMES = {
    "A673_Hic": "A673_Tad.bedpe.gz",
    "Caco2_Hic": "Caco2_Tad.bedpe.gz",
    "Calu3_Hic": "Calu3_Tad.bedpe.gz",
    "CH12LX_Hic": "CH12LX_Tad.bedpe.gz",
    "GM12878_Hic": "Gm12878_Tad.bedpe.gz",
    "GM23248_Hic": "GM23248_Tad.bedpe.gz",
    "HepG2_Hic": "HepG2_Tad.bedpe.gz",
    "IMR90_Hic": "IMR90_Tad.bedpe.gz",
    "MCF10A_Hic": "MCF10A_Tad.bedpe.gz",
    "Mcf7_Hic": "Mcf7_Tad.bedpe.gz",
    "OCILY7_Hic": "OCILY7_Tad.bedpe.gz",
    "Panc1_Hic": "Panc1_Tad.bedpe.gz",
    "PC3_Hic": "PC3_Tad.bedpe.gz",
    "PC9_Hic": "PC9_Tad.bedpe.gz",
    "T47D_Hic": "T47D_Tad.bedpe.gz",
}

LOOP_BEDPE_FILENAMES = [
    "A673_Loop.bedpe.gz",
    "Caco2_Loop.bedpe.gz",
    "Calu3_Loop.bedpe.gz",
    "CH12LX_Loop.bedpe.gz",
    "Gm12878_Loop.bedpe.gz",
    "GM23248_Loop.bedpe.gz",
    "Hct116_Loop.bedpe.gz",
    "HepG2_Loop.bedpe.gz",
    "IMR90_Loop.bedpe.gz",
    "MCF10A_Loop.bedpe.gz",
    "Mcf7_Loop.bedpe.gz",
    "OCILY7_Loop.bedpe.gz",
    "Panc1_Loop.bedpe.gz",
    "PC3_Loop.bedpe.gz",
    "PC9_Loop.bedpe.gz",
    "T47D_Loop.bedpe.gz",
]


def dense2tag(matrix: np.ndarray):
    matrix = np.triu(matrix)
    matrix = np.maximum(matrix, 0)
    tag_len = np.sum(matrix).astype(np.int64)
    if tag_len == 0:
        return np.zeros((0, 2), dtype=np.int32), 0

    tag_mat = np.zeros((tag_len, 2), dtype=np.int32)
    coo_mat = coo_matrix(matrix)
    row, col, data = coo_mat.row, coo_mat.col, coo_mat.data.astype(np.int64)
    start_idx = 0
    for idx in range(len(row)):
        end_idx = start_idx + data[idx]
        tag_mat[start_idx:end_idx, :] = (row[idx], col[idx])
        start_idx = end_idx
    return tag_mat, tag_len


def tag2dense(tag: np.ndarray, nsize: int):
    if tag.shape[0] == 0:
        return np.zeros((nsize, nsize), dtype=np.int32)

    coo_data, data = np.unique(tag, axis=0, return_counts=True)
    row, col = coo_data[:, 0], coo_data[:, 1]
    dense_mat = coo_matrix((data, (row, col)), shape=(nsize, nsize)).toarray()
    dense_mat = dense_mat + np.triu(dense_mat, k=1).T
    return dense_mat


def downsampling_deephic(matrix: np.ndarray, down_ratio: float):
    matrix_int = matrix.astype(np.int32)
    tag_mat, tag_len = dense2tag(matrix_int)
    if tag_len == 0 or down_ratio <= 1:
        return matrix_int.copy()

    n_samples = max(1, int(tag_len / down_ratio))
    sample_idx = np.random.choice(tag_len, n_samples, replace=False)
    return tag2dense(tag_mat[sample_idx], matrix.shape[0])


def encode_sequence_window(chr_arr, start_bp: int, end_bp: int, one_hot_table: np.ndarray):
    target_len = end_bp - start_bp
    encoded = np.zeros((target_len, 4), dtype=np.float32)
    actual_start = max(0, start_bp)
    actual_end = min(len(chr_arr), end_bp)
    if actual_start >= actual_end:
        return encoded

    offset = actual_start - start_bp
    segment = chr_arr[actual_start:actual_end]
    if isinstance(segment, np.ndarray):
        encoded[offset : offset + len(segment)] = one_hot_table[segment]
    else:
        segment_arr = np.frombuffer(str(segment).encode("ascii"), dtype=np.uint8)
        encoded[offset : offset + len(segment_arr)] = one_hot_table[segment_arr]
    return encoded


def reverse_complement_one_hot(seq: np.ndarray):
    return np.flip(seq, axis=0).copy()[:, [3, 2, 1, 0]]


def normalize_hic_matrix(hic_data):
    if isinstance(hic_data, dict) and "target" in hic_data:
        hic_data = hic_data["target"]
    if hasattr(hic_data, "float"):
        hic_data = hic_data.float().numpy()
    elif hasattr(hic_data, "numpy"):
        hic_data = hic_data.numpy()
    hic_target = np.asarray(hic_data, dtype=np.float32)
    hic_target = np.nan_to_num(hic_target, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    hic_target = np.clip(hic_target, a_min=0.0, a_max=None)
    hic_target = np.rint(hic_target).astype(np.int32)
    np.fill_diagonal(hic_target, 0)
    return hic_target


def hic_to_tensor(hic_matrix: np.ndarray):
    hic_tensor = torch.from_numpy(hic_matrix).float().log1p()
    hic_max = hic_tensor.max()
    if hic_max > 0:
        hic_tensor = hic_tensor / hic_max
    return hic_tensor.unsqueeze(0)


def parse_chroms(value: str):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str):
    if device_arg.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_arg)
    return torch.device("cpu")


def safe_torch_load(path: str, map_location="cpu", weights_only=True):
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_matching_state_dict(model, checkpoint_path: str, *, label: str = "checkpoint"):
    if not checkpoint_path:
        return None
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"{label} not found: {checkpoint_path}")

    state_dict = safe_torch_load(checkpoint_path, map_location="cpu", weights_only=True)
    model_state = model.state_dict()
    matched = {}
    skipped = []
    for key, value in state_dict.items():
        norm_key = key[7:] if key.startswith("module.") else key
        if norm_key in model_state and model_state[norm_key].shape == value.shape:
            matched[norm_key] = value
        else:
            skipped.append(norm_key)

    result = model.load_state_dict(matched, strict=False)
    print(
        f"Loaded {len(matched)} tensors from {checkpoint_path} "
        f"({label}); missing={len(result.missing_keys)}, skipped={len(skipped)}"
    )
    return result


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def default_tad_bedpe_map(bedpe_dir: str = DEFAULT_BEDPE_DIR):
    return {
        cell: str(Path(bedpe_dir) / filename)
        for cell, filename in TAD_BEDPE_FILENAMES.items()
    }


def default_loop_paths(bedpe_dir: str = DEFAULT_BEDPE_DIR):
    return [str(Path(bedpe_dir) / filename) for filename in LOOP_BEDPE_FILENAMES]


def load_path_map(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        if path.endswith(".json"):
            data = json.load(handle)
            return {str(key): str(value) for key, value in data.items()}

        mapping = {}
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", "\t").split()
            if len(parts) < 2:
                raise ValueError(f"Invalid map line in {path!r}: {line!r}")
            mapping[parts[0]] = parts[1]
    return mapping


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def pearson_corr_np(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return 0.0
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = math.sqrt(float(np.sum(x_centered**2) * np.sum(y_centered**2)))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(x_centered * y_centered) / denom)


def pearson_corr_loss(pred: torch.Tensor, target: torch.Tensor):
    pred = pred.reshape(pred.shape[0] * pred.shape[1], pred.shape[2])
    target = target.reshape(target.shape[0] * target.shape[1], target.shape[2])
    pred_centered = pred - pred.mean(dim=1, keepdim=True)
    target_centered = target - target.mean(dim=1, keepdim=True)
    covariance = (pred_centered * target_centered).sum(dim=1)
    pred_var = (pred_centered**2).sum(dim=1) + 1e-8
    target_var = (target_centered**2).sum(dim=1) + 1e-8
    corr = covariance / torch.sqrt(pred_var * target_var)
    return 1.0 - corr.mean()


def parse_loops(bedpe_path: str, chroms: Sequence[str] = ALL_CHROMS):
    loops = {chrom: [] for chrom in chroms}
    open_fn = gzip.open if str(bedpe_path).endswith(".gz") else open
    with open_fn(bedpe_path, "rt") as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            chrom1, start1, end1, chrom2, start2, end2 = parts[:6]
            if chrom1 != chrom2 or chrom1 not in loops:
                continue
            center1 = (int(start1) + int(end1)) // 2
            center2 = (int(start2) + int(end2)) // 2
            loops[chrom1].append((center1 // HIC_RESOLUTION, center2 // HIC_RESOLUTION))
    return loops


def cell_prefix_from_loop_path(loop_path: str):
    basename = os.path.basename(loop_path)
    return basename.split("_Loop")[0]


def find_matching_hic_dir(hic_root: str, cell_prefix: str):
    if not os.path.isdir(hic_root):
        return None
    expected = f"{cell_prefix}_Hic".lower()
    for dirname in os.listdir(hic_root):
        if dirname.lower() == expected:
            return os.path.join(hic_root, dirname)
    return None


def filter_existing(paths: Iterable[str]):
    return [path for path in paths if os.path.exists(path)]
