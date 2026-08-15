import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from data_processing.common import (
    DNA_BP_WINDOW,
    HIC_RESOLUTION,
    downsampling_deephic,
    encode_sequence_window,
    hic_to_tensor,
    normalize_hic_matrix,
    reverse_complement_one_hot,
)
from data_processing.dna_loader import build_one_hot_table, load_chr


class GeneralHiCDataset(Dataset):
    def __init__(
        self,
        hic_root: str,
        seq_root: str,
        cell_lines_list,
        chroms_list,
        mode: str = "train",
        rc_prob: float = 0.5,
        max_samples=None,
        sample_stride: int = 1,
    ) -> None:
        self.hic_root = hic_root
        self.seq_root = seq_root
        self.mode = mode
        self.rc_prob = rc_prob
        self.sample_stride = max(1, sample_stride)
        self.one_hot_table = build_one_hot_table()
        self.samples = self._collect_samples(cell_lines_list, chroms_list)
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        self.seq_objects = {
            chrom: load_chr(chrom, seq_root, mmap=True)
            for chrom in set(chroms_list)
        }

    def _collect_samples(self, cell_lines_list, chroms_list):
        samples = []
        for cell in cell_lines_list:
            cell_dir = os.path.join(self.hic_root, cell)
            if not os.path.isdir(cell_dir):
                continue
            for chrom in chroms_list:
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
                    if os.path.exists(pt_path):
                        samples.append(
                            {
                                "path": pt_path,
                                "start": start_idx,
                                "chrom": chrom,
                                "cell": cell,
                            }
                        )
        return samples

    def __len__(self):
        return len(self.samples)

    def load_raw_data(self, idx: int, jitter: int = 0):
        info = self.samples[idx]
        hic_target = normalize_hic_matrix(
            torch.load(info["path"], map_location="cpu", weights_only=False)
        )
        start_bp = int(info["start"]) * HIC_RESOLUTION + jitter
        end_bp = start_bp + DNA_BP_WINDOW
        seq_data = encode_sequence_window(
            self.seq_objects[info["chrom"]],
            start_bp,
            end_bp,
            self.one_hot_table,
        )
        return seq_data, hic_target, info

    def __getitem__(self, idx):
        jitter = random.randint(-200, 200) if self.mode == "train" else 0
        seq_data, hic_target, _ = self.load_raw_data(idx, jitter)

        if self.mode == "train" and random.random() < self.rc_prob:
            hic_target = np.flip(hic_target, axis=(0, 1)).copy()
            seq_data = reverse_complement_one_hot(seq_data)

        if self.mode == "train":
            k = random.uniform(2, 25)
            hic_low = downsampling_deephic(hic_target, float(k**2))
        else:
            hic_low = downsampling_deephic(hic_target, 16.0)

        seq_tensor = torch.from_numpy(seq_data).float().transpose(0, 1)
        target_tensor = torch.from_numpy(hic_target).float().log1p()
        return seq_tensor, hic_to_tensor(hic_low), target_tensor
