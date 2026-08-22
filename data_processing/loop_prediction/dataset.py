import os
import random
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from data_processing.common import (
    DNA_BP_WINDOW,
    HIC_RESOLUTION,
    NUM_HIC_BINS,
    cell_prefix_from_loop_path,
    downsampling_deephic,
    encode_sequence_window,
    find_matching_hic_dir,
    hic_to_tensor,
    normalize_hic_matrix,
    parse_loops,
    reverse_complement_one_hot,
)
from data_processing.dna_loader import build_one_hot_table, load_chr


class DNA2LoopDataset(Dataset):
    def __init__(
        self,
        hic_root: str,
        seq_root: str,
        loop_paths: List[str],
        chroms: Sequence[str],
        mode: str = "train",
        rc_prob: float = 0.5,
        downsample_hic: bool = False,
        downsample_k_range: tuple = (2.0, 25.0),
        max_samples: Optional[int] = None,
        sample_stride: int = 1,
    ) -> None:
        super().__init__()
        self.hic_root = hic_root
        self.seq_root = seq_root
        self.chroms = list(chroms)
        self.mode = mode
        self.rc_prob = rc_prob
        self.downsample_hic = downsample_hic
        self.downsample_k_range = downsample_k_range
        self.max_samples = max_samples
        self.sample_stride = max(1, sample_stride)

        self.one_hot_table = build_one_hot_table()
        self.all_loops: Dict[str, Dict[str, List[tuple]]] = {}
        self.cell_hic_dirs: Dict[str, str] = {}

        for loop_path in loop_paths:
            if not os.path.exists(loop_path):
                continue
            cell_prefix = cell_prefix_from_loop_path(loop_path)
            hic_dir = find_matching_hic_dir(self.hic_root, cell_prefix)
            if hic_dir and os.path.isdir(hic_dir):
                self.cell_hic_dirs[cell_prefix] = hic_dir
                self.all_loops[cell_prefix] = parse_loops(loop_path, chroms=self.chroms)

        self.chr_seqs = {chrom: load_chr(chrom, seq_root, mmap=True) for chrom in self.chroms}
        self.samples = self._collect_samples()
        if self.max_samples is not None:
            random.shuffle(self.samples)
            self.samples = self.samples[: self.max_samples]

    def _collect_samples(self):
        samples = []
        for cell_prefix, cell_dir in self.cell_hic_dirs.items():
            for chrom in self.chroms:
                chrom_dir = os.path.join(cell_dir, chrom)
                meta_path = os.path.join(chrom_dir, "meta.txt")
                if not os.path.exists(meta_path):
                    continue
                with open(meta_path, "r", encoding="utf-8") as handle:
                    starts = [int(line.strip()) for line in handle if line.strip()]
                for idx, start_idx in enumerate(starts):
                    if idx % self.sample_stride != 0:
                        continue
                    pt_path = os.path.join(chrom_dir, f"{start_idx}.pt")
                    if not os.path.exists(pt_path):
                        continue
                    samples.append(
                        {
                            "cell": cell_prefix,
                            "chrom": chrom,
                            "start_idx": start_idx,
                            "pt_path": pt_path,
                        }
                    )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_hic_high_depth(self, pt_path: str):
        return normalize_hic_matrix(torch.load(pt_path, map_location="cpu", weights_only=False))

    def _load_targets(self, cell: str, chrom: str, start_idx: int):
        start_bin = start_idx
        end_bin = start_bin + NUM_HIC_BINS
        target = np.zeros((NUM_HIC_BINS, NUM_HIC_BINS), dtype=np.int64)
        for bin1, bin2 in self.all_loops.get(cell, {}).get(chrom, []):
            if start_bin <= bin1 < end_bin and start_bin <= bin2 < end_bin:
                target[bin1 - start_bin, bin2 - start_bin] = 1
                target[bin2 - start_bin, bin1 - start_bin] = 1
        return target

    def load_raw_data(self, index: int, jitter: int = 0):
        sample = self.samples[index]
        start_bp = sample["start_idx"] * HIC_RESOLUTION + jitter
        end_bp = start_bp + DNA_BP_WINDOW
        seq = encode_sequence_window(
            self.chr_seqs[sample["chrom"]],
            start_bp,
            end_bp,
            self.one_hot_table,
        )
        hic_target = self._load_hic_high_depth(sample["pt_path"])
        target_loop = self._load_targets(sample["cell"], sample["chrom"], sample["start_idx"])
        return seq, hic_target, target_loop, sample

    def __getitem__(self, index: int):
        jitter = random.randint(-200, 200) if self.mode == "train" else 0
        seq, hic_target, target_loop, sample = self.load_raw_data(index, jitter=jitter)

        if self.mode == "train" and random.random() < self.rc_prob:
            seq = reverse_complement_one_hot(seq)
            hic_target = np.flip(hic_target, axis=(0, 1)).copy()
            target_loop = np.flip(target_loop, axis=(0, 1)).copy()

        if self.downsample_hic:
            if self.mode == "train":
                k = random.uniform(*self.downsample_k_range)
                hic_target = downsampling_deephic(hic_target, float(k**2))
            elif self.mode == "val":
                hic_target = downsampling_deephic(hic_target, 16.0)

        seq_tensor = torch.from_numpy(seq).float().transpose(0, 1)
        target_loop_tensor = torch.from_numpy(target_loop).long()
        return (
            seq_tensor,
            hic_to_tensor(hic_target),
            target_loop_tensor,
            sample["chrom"],
            int(sample["start_idx"] * HIC_RESOLUTION),
        )
