#!/usr/bin/env python3
"""
Stage-1 ChromSeek pretraining: DNA -> 1D multi-omics at 200 bp resolution.

This script is the standalone driver version of the original
`cc6_200bp.ipynb` training workflow.  It reuses the model definition from
`utils/cc6_200bp.py` and keeps the original core behavior:

  - DNA input length: 448,000 bp
  - output resolution: 200 bp, 2,240 bins
  - cell-specific model with cell embeddings
  - 15 one-dimensional omics tasks
  - random interval-level 95/5 train/validation split
  - masked MSE loss on the middle 80% of bins
  - validation mean task-wise PCC selects best checkpoint
  - best checkpoint filename: best_model_448k_200bp.pth
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pyfaidx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import pearsonr, spearmanr
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.cc6_200bp import CellSpecificOmicsModel_448k  # noqa: E402

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional runtime dependency
    SummaryWriter = None


MASTER_TASK_LIST = [
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

DNA_TO_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}


class NullWriter:
    def add_scalar(self, *_args: Any, **_kwargs: Any) -> None:
        return

    def close(self) -> None:
        return


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dna_to_one_hot(seq_str: str) -> torch.Tensor:
    seq_indices = [DNA_TO_INDEX.get(base, -1) for base in seq_str.upper()]
    seq_tensor = torch.tensor(seq_indices, dtype=torch.long)
    n_mask = seq_tensor == -1
    seq_tensor_safe = seq_tensor.clone()
    seq_tensor_safe[n_mask] = 0
    one_hot = F.one_hot(seq_tensor_safe, num_classes=4).float()
    one_hot[n_mask] = 0.25
    return one_hot.transpose(0, 1)


class MultiCellGenomicDataset(Dataset):
    def __init__(
        self,
        fasta_path: str,
        global_data_all_cells: dict[str, dict[str, dict[str, np.ndarray] | None]],
        cell_map: dict[str, int],
        task_names: list[str],
        intervals: list[tuple[str, int, int, str]],
        seq_length: int,
        target_bins: int,
        bin_size: int,
        augment_jitter: bool = False,
        jitter_bp: int = 20,
        smooth_kernel_np: np.ndarray | None = None,
        valid_bin_start: int = 0,
        valid_bin_end: int | None = None,
    ) -> None:
        super().__init__()
        self.fasta_path = fasta_path
        self.global_data = global_data_all_cells
        self.cell_map = cell_map
        self.task_names = task_names
        self.num_tasks = len(task_names)
        self.intervals = intervals

        self.seq_length = seq_length
        self.target_bins = target_bins
        self.bin_size = bin_size
        self.target_length_bp = target_bins * bin_size
        self.padding = (seq_length - self.target_length_bp) // 2

        self.augment_jitter = augment_jitter
        self.jitter_bp = jitter_bp
        self.valid_bin_start = valid_bin_start
        self.valid_bin_end = valid_bin_end if valid_bin_end is not None else target_bins

        self.fasta: pyfaidx.Fasta | None = None
        self.smooth_kernel_tensor: torch.Tensor | None = None
        self.smooth_padding = 0
        if smooth_kernel_np is not None:
            kernel = torch.tensor(smooth_kernel_np, dtype=torch.float32)
            self.smooth_kernel_tensor = kernel.view(1, 1, len(kernel))
            self.smooth_padding = (len(kernel) - 1) // 2

        if not self.intervals:
            warnings.warn("Dataset received an empty interval list.", stacklevel=2)

    def __len__(self) -> int:
        return len(self.intervals)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.fasta is None:
            self.fasta = pyfaidx.Fasta(self.fasta_path)

        chrom, start, cell_id, cell_name = self.intervals[idx]

        jitter_offset = 0
        if self.augment_jitter:
            jitter_offset = random.randint(-self.jitter_bp, self.jitter_bp)

        start_bp_dna = start + jitter_offset
        end_bp_dna = start_bp_dna + self.seq_length

        try:
            if start_bp_dna < 0:
                raise ValueError("negative DNA start after jitter")
            seq_str = self.fasta[chrom][start_bp_dna:end_bp_dna].seq
            if len(seq_str) != self.seq_length:
                raise ValueError("short DNA slice after jitter")
        except Exception:
            start_bp_dna = start
            end_bp_dna = start_bp_dna + self.seq_length
            try:
                seq_str = self.fasta[chrom][start_bp_dna:end_bp_dna].seq
                if len(seq_str) != self.seq_length:
                    seq_str = "N" * self.seq_length
            except Exception:
                seq_str = "N" * self.seq_length

        sequence = dna_to_one_hot(seq_str)

        target_start_bp = start + self.padding
        start_bin = target_start_bp // self.bin_size
        end_bin = start_bin + self.target_bins

        target_signals = []
        target_masks = []
        for task in self.task_names:
            task_data = self.global_data[cell_name].get(task)
            if task_data is not None and chrom in task_data:
                signal = task_data[chrom][start_bin:end_bin].astype(np.float32, copy=False)
                if len(signal) < self.target_bins:
                    pad = np.zeros(self.target_bins - len(signal), dtype=np.float32)
                    signal = np.concatenate([signal, pad])
                else:
                    signal = signal[: self.target_bins]
                initial_mask = np.ones_like(signal, dtype=np.float32)
            else:
                signal = np.zeros(self.target_bins, dtype=np.float32)
                initial_mask = np.zeros(self.target_bins, dtype=np.float32)

            final_mask = np.zeros(self.target_bins, dtype=np.float32)
            final_mask[self.valid_bin_start : self.valid_bin_end] = 1.0
            final_mask *= initial_mask

            target_signals.append(signal)
            target_masks.append(final_mask)

        target_raw = torch.tensor(np.stack(target_signals, axis=0), dtype=torch.float32)
        target_mask = torch.tensor(np.stack(target_masks, axis=0), dtype=torch.float32)

        target_smoothed = target_raw
        if self.smooth_kernel_tensor is not None:
            kernel = self.smooth_kernel_tensor.repeat(self.num_tasks, 1, 1)
            target_smoothed = F.conv1d(
                target_raw.unsqueeze(0),
                kernel,
                padding=self.smooth_padding,
                groups=self.num_tasks,
            ).squeeze(0)

        return {
            "sequence": sequence,
            "target": target_smoothed,
            "target_raw": target_raw,
            "target_mask": target_mask,
            "cell_id": torch.tensor(cell_id, dtype=torch.long),
            "cell_name": cell_name,
            "chrom": chrom,
            "start": start,
        }


def load_precomputed_omics(
    precomputed_dir: str,
    cell_names: list[str],
    task_names: list[str],
) -> dict[str, dict[str, dict[str, np.ndarray] | None]]:
    global_data: dict[str, dict[str, dict[str, np.ndarray] | None]] = {}
    for cell_name in tqdm(cell_names, desc="Loading cell omics"):
        global_data[cell_name] = {}
        for task_name in task_names:
            npz_path = Path(precomputed_dir) / cell_name / f"{task_name}.npz"
            if not npz_path.exists():
                global_data[cell_name][task_name] = None
                continue
            try:
                with np.load(npz_path) as npz_file:
                    global_data[cell_name][task_name] = {
                        chrom: npz_file[chrom].astype(np.float32, copy=False)
                        for chrom in npz_file.files
                    }
            except Exception as exc:
                print(f"[WARN] failed to load {npz_path}: {exc}")
                global_data[cell_name][task_name] = None
    return global_data


def build_intervals(
    fasta_path: str,
    chroms: list[str],
    seq_length: int,
    cell_map: dict[str, int],
    limit_intervals: int | None = None,
) -> list[tuple[str, int, int, str]]:
    intervals: list[tuple[str, int, int, str]] = []
    fasta = pyfaidx.Fasta(fasta_path)
    try:
        for chrom in tqdm(chroms, desc="Scanning chromosomes"):
            if chrom not in fasta:
                print(f"[WARN] {chrom} not found in FASTA, skipped.")
                continue
            chrom_len = len(fasta[chrom])
            for start_bp in range(0, chrom_len - seq_length, seq_length):
                for cell_name, cell_id in cell_map.items():
                    intervals.append((chrom, start_bp, cell_id, cell_name))
                if limit_intervals is not None and len(intervals) >= limit_intervals:
                    return intervals[:limit_intervals]
    finally:
        fasta.close()
    return intervals


def masked_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    loss_fn: nn.Module,
) -> torch.Tensor:
    raw_loss = loss_fn(pred, target)
    masked_loss = raw_loss * mask
    return masked_loss.sum() / mask.sum().clamp(min=1.0)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    scaler: GradScaler,
    amp_enabled: bool,
    epoch: int,
    writer: Any,
    max_steps: int | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    steps = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)
    for step, batch in enumerate(pbar, start=1):
        seq = batch["sequence"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        mask = batch["target_mask"].to(device, non_blocking=True)
        cell_id = batch["cell_id"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp_enabled, dtype=torch.float16):
            pred = model(seq, cell_id)
            loss = masked_mse_loss(pred, target, mask, loss_fn)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_item = float(loss.item())
        total_loss += loss_item
        steps += 1
        pbar.set_postfix(loss=f"{loss_item:.4f}")

        if step % 50 == 0:
            writer.add_scalar("Loss/train_batch", loss_item, (epoch - 1) * len(loader) + step)
        if max_steps is not None and step >= max_steps:
            break

    return total_loss / max(1, steps)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    task_names: list[str],
    amp_enabled: bool,
    max_steps: int | None = None,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    steps = 0
    preds_per_task: list[list[np.ndarray]] = [[] for _ in task_names]
    reals_per_task: list[list[np.ndarray]] = [[] for _ in task_names]

    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation", leave=False)
        for step, batch in enumerate(pbar, start=1):
            seq = batch["sequence"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            target_raw = batch["target_raw"].to(device, non_blocking=True)
            mask = batch["target_mask"].to(device, non_blocking=True)
            cell_id = batch["cell_id"].to(device, non_blocking=True)

            with autocast(enabled=amp_enabled, dtype=torch.float16):
                pred = model(seq, cell_id)
                loss = masked_mse_loss(pred, target, mask, loss_fn)

            total_loss += float(loss.item())
            steps += 1

            pred_np = pred.float().cpu().numpy()
            raw_np = target_raw.float().cpu().numpy()
            mask_np = mask.float().cpu().numpy()

            for task_idx in range(len(task_names)):
                valid = mask_np[:, task_idx, :] > 0
                if valid.sum() < 50:
                    continue
                preds_per_task[task_idx].append(pred_np[:, task_idx, :][valid])
                reals_per_task[task_idx].append(raw_np[:, task_idx, :][valid])

            if max_steps is not None and step >= max_steps:
                break

    pcc_values = []
    scc_values = []
    for task_idx, task_name in enumerate(task_names):
        if not preds_per_task[task_idx]:
            continue
        preds = np.concatenate(preds_per_task[task_idx])
        reals = np.concatenate(reals_per_task[task_idx])
        if preds.size < 50 or np.std(preds) < 1e-6 or np.std(reals) < 1e-6:
            print(f"[WARN] skip metric for {task_name}: insufficient variance/data")
            continue

        pcc = pearsonr(preds, reals)[0]
        scc = spearmanr(preds, reals)[0]
        if np.isfinite(pcc):
            pcc_values.append(float(pcc))
        if np.isfinite(scc):
            scc_values.append(float(scc))

    mean_pcc = float(np.mean(pcc_values)) if pcc_values else 0.0
    mean_scc = float(np.mean(scc_values)) if scc_values else 0.0
    avg_loss = total_loss / max(1, steps)
    return mean_pcc, mean_scc, avg_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage-1 ChromSeek pretraining: DNA -> 1D multi-omics."
    )
    parser.add_argument("--fasta_path", default="/mnt/nfs/jyzhu/dataset/sequence/hg38.fa")
    parser.add_argument(
        "--precomputed_dir",
        default="/mnt/nfs/jyzhu/dataset/1d_omic/precomputed_200bp_bins_normalized_cpu",
    )
    parser.add_argument("--output_dir", default="runs/stage1_448k_200bp")
    parser.add_argument("--best_name", default="best_model_448k_200bp.pth")
    parser.add_argument("--seq_length", type=int, default=448_000)
    parser.add_argument("--target_bins", type=int, default=2_240)
    parser.add_argument("--bin_size", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=40)
    parser.add_argument("--val_num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_ratio", type=float, default=0.95)
    parser.add_argument("--jitter_bp", type=int, default=20)
    parser.add_argument("--no_jitter", action="store_true")
    parser.add_argument("--chroms", nargs="+", default=[f"chr{i}" for i in range(1, 23)])
    parser.add_argument("--cell_names", nargs="+", default=None)
    parser.add_argument("--task_names", nargs="+", default=MASTER_TASK_LIST)
    parser.add_argument("--limit_intervals", type=int, default=None)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--max_val_steps", type=int, default=None)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--no_tensorboard", action="store_true")
    parser.add_argument("--no_dataparallel", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.seq_length != args.target_bins * args.bin_size:
        raise ValueError(
            "This stage-1 driver expects seq_length == target_bins * bin_size "
            f"({args.seq_length} vs {args.target_bins * args.bin_size})."
        )
    if not os.path.exists(args.fasta_path):
        raise FileNotFoundError(f"FASTA not found: {args.fasta_path}")
    if not os.path.isdir(args.precomputed_dir):
        raise FileNotFoundError(f"precomputed_dir not found: {args.precomputed_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / args.best_name
    last_path = output_dir / "last_model.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = (device.type == "cuda") and (not args.no_amp)

    cell_names = args.cell_names
    if cell_names is None:
        cell_names = sorted(
            d.name for d in Path(args.precomputed_dir).iterdir() if d.is_dir()
        )
    if not cell_names:
        raise RuntimeError(f"No cell subdirectories found in {args.precomputed_dir}")

    cell_map = {name: idx for idx, name in enumerate(cell_names)}
    valid_bin_start = int(args.target_bins * 0.1)
    valid_bin_end = args.target_bins - valid_bin_start

    print(f"Device: {device}")
    print(f"Cells: {len(cell_map)}")
    print(f"Tasks: {len(args.task_names)}")
    print(f"Input: {args.seq_length} bp")
    print(f"Output: {args.target_bins} bins at {args.bin_size} bp")
    print(f"Valid bins: [{valid_bin_start}, {valid_bin_end})")

    global_data = load_precomputed_omics(args.precomputed_dir, cell_names, args.task_names)
    intervals = build_intervals(
        args.fasta_path,
        args.chroms,
        args.seq_length,
        cell_map,
        limit_intervals=args.limit_intervals,
    )
    if not intervals:
        raise RuntimeError("No intervals were generated.")

    random.shuffle(intervals)
    split_idx = int(len(intervals) * args.split_ratio)
    train_intervals = intervals[:split_idx]
    val_intervals = intervals[split_idx:]
    print(f"Intervals: total={len(intervals)}, train={len(train_intervals)}, val={len(val_intervals)}")

    smooth_kernel = np.array([0.1, 0.8, 0.1], dtype=np.float32)
    train_dataset = MultiCellGenomicDataset(
        args.fasta_path,
        global_data,
        cell_map,
        args.task_names,
        train_intervals,
        args.seq_length,
        args.target_bins,
        args.bin_size,
        augment_jitter=not args.no_jitter,
        jitter_bp=args.jitter_bp,
        smooth_kernel_np=smooth_kernel,
        valid_bin_start=valid_bin_start,
        valid_bin_end=valid_bin_end,
    )
    val_dataset = MultiCellGenomicDataset(
        args.fasta_path,
        global_data,
        cell_map,
        args.task_names,
        val_intervals,
        args.seq_length,
        args.target_bins,
        args.bin_size,
        augment_jitter=False,
        jitter_bp=0,
        smooth_kernel_np=smooth_kernel,
        valid_bin_start=valid_bin_start,
        valid_bin_end=valid_bin_end,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.val_num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = CellSpecificOmicsModel_448k(
        num_cells=len(cell_map),
        embed_dim=args.embed_dim,
        num_tasks=len(args.task_names),
        seq_len=args.seq_length,
        encoder_channels=[64, 128, 256, 384],
    ).to(device)

    if torch.cuda.device_count() > 1 and not args.no_dataparallel:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss(reduction="none")
    scaler = GradScaler(enabled=amp_enabled)

    writer: Any = NullWriter()
    if not args.no_tensorboard and SummaryWriter is not None:
        log_dir = output_dir / f"tensorboard_{time.strftime('%Y%m%d-%H%M%S')}"
        writer = SummaryWriter(log_dir=str(log_dir))
        print(f"TensorBoard: tensorboard --logdir {log_dir}")

    best_pcc = -1.0
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            scaler,
            amp_enabled,
            epoch,
            writer,
            max_steps=args.max_train_steps,
        )
        val_pcc, val_scc, val_loss = evaluate(
            model,
            val_loader,
            loss_fn,
            device,
            args.task_names,
            amp_enabled,
            max_steps=args.max_val_steps,
        )

        writer.add_scalar("Loss/train_epoch", train_loss, epoch)
        writer.add_scalar("Loss/validation", val_loss, epoch)
        writer.add_scalar("PCC_Raw/validation_mean", val_pcc, epoch)
        writer.add_scalar("SCC_Raw/validation_mean", val_scc, epoch)

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.5f} | "
            f"val_loss={val_loss:.5f} | val_pcc={val_pcc:.5f} | val_scc={val_scc:.5f}"
        )

        torch.save(model.state_dict(), last_path)
        if val_pcc > best_pcc:
            best_pcc = val_pcc
            torch.save(model.state_dict(), best_path)
            print(f"[*] Saved new best checkpoint: {best_path} (val_pcc={best_pcc:.5f})")

    writer.close()
    print(f"Training complete. Best validation PCC: {best_pcc:.5f}")
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
