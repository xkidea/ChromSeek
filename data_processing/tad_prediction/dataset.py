import gzip
import os
import random
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from data_processing.common import (
    BIN_SIZE,
    DNA_BP_WINDOW,
    HIC_RESOLUTION,
    NUM_1KB_BINS,
    default_tad_bedpe_map,
    downsampling_deephic,
    encode_sequence_window,
    hic_to_tensor,
    normalize_hic_matrix,
    reverse_complement_one_hot,
)
from data_processing.dna_loader import build_one_hot_table, load_chr


class DNA2TadDataset(Dataset):
    def __init__(
        self,
        hic_root: str,
        seq_root: str,
        cell_lines_list: Sequence[str],
        chroms: Optional[Sequence[str]] = None,
        chroms_list: Optional[Sequence[str]] = None,
        mode: str = "train",
        rc_prob: float = 0.5,
        augment_rc: Optional[bool] = None,
        max_samples: Optional[int] = None,
        sample_stride: int = 1,
        downsample_hic: bool = True,
        use_downsample: Optional[bool] = None,
        tad_bedpe_map: Optional[Mapping[str, str]] = None,
        downsample_k_range: tuple = (1.0, 10.0),
    ) -> None:
        super().__init__()
        self.hic_root = hic_root
        self.seq_root = seq_root
        self.chroms = list(chroms if chroms is not None else chroms_list or [])
        self.mode = mode
        self.rc_prob = 0.5 if augment_rc is True else 0.0 if augment_rc is False else rc_prob
        self.max_samples = max_samples
        self.sample_stride = max(1, sample_stride)
        self.downsample_hic = downsample_hic if use_downsample is None else use_downsample
        self.downsample_k_range = downsample_k_range
        self.cell_lines_list = list(cell_lines_list)
        self.tad_bedpe_map = dict(tad_bedpe_map or default_tad_bedpe_map())

        self.one_hot_table = build_one_hot_table()
        self.tad_labels: Dict[str, Dict[str, np.ndarray]] = self._load_tad_labels()
        self.chr_seqs = {chrom: load_chr(chrom, seq_root, mmap=True) for chrom in self.chroms}
        self.samples = self._collect_samples()
        if self.max_samples is not None:
            self.samples = self.samples[: self.max_samples]

    def _load_tad_labels(self) -> Dict[str, Dict[str, np.ndarray]]:
        all_tad_arrays = {}
        for cell in self.cell_lines_list:
            bedpe_path = self.tad_bedpe_map.get(cell)
            tad_arrays = {chrom: np.zeros(300000, dtype=np.int64) for chrom in self.chroms}
            if bedpe_path and os.path.exists(bedpe_path):
                open_fn = gzip.open if str(bedpe_path).endswith(".gz") else open
                with open_fn(bedpe_path, "rt") as handle:
                    for line in handle:
                        if not line.startswith("chr"):
                            continue
                        parts = line.strip().split()
                        if len(parts) < 3:
                            continue
                        chrom = parts[0]
                        if chrom not in tad_arrays:
                            continue
                        start_bin = int(parts[1]) // BIN_SIZE
                        end_bin = int(parts[2]) // BIN_SIZE
                        if start_bin < len(tad_arrays[chrom]):
                            tad_arrays[chrom][start_bin] = 1
                        if end_bin < len(tad_arrays[chrom]):
                            tad_arrays[chrom][end_bin] = 1
            all_tad_arrays[cell] = tad_arrays
        return all_tad_arrays

    def _collect_samples(self):
        samples = []
        for cell in self.cell_lines_list:
            cell_dir = os.path.join(self.hic_root, cell)
            if not os.path.isdir(cell_dir):
                continue
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
                            "cell": cell,
                            "chrom": chrom,
                            "start_idx": start_idx,
                            "start_bp": start_idx * HIC_RESOLUTION,
                            "pt_path": pt_path,
                        }
                    )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_hic_high_depth(self, pt_path: str):
        return normalize_hic_matrix(torch.load(pt_path, map_location="cpu", weights_only=False))

    def _load_targets(self, cell: str, chrom: str, start_bp: int):
        start_bin = start_bp // BIN_SIZE
        end_bin = start_bin + NUM_1KB_BINS
        if cell in self.tad_labels and chrom in self.tad_labels[cell]:
            labels = self.tad_labels[cell][chrom][start_bin:end_bin].copy()
        else:
            labels = np.zeros(NUM_1KB_BINS, dtype=np.int64)
        if labels.shape[0] < NUM_1KB_BINS:
            padded = np.zeros(NUM_1KB_BINS, dtype=np.int64)
            padded[: labels.shape[0]] = labels
            labels = padded
        return labels

    def load_raw_data(self, index: int, jitter: int = 0):
        sample = self.samples[index]
        start_bp = sample["start_bp"] + jitter
        end_bp = start_bp + DNA_BP_WINDOW

        hic_target = self._load_hic_high_depth(sample["pt_path"])
        seq = encode_sequence_window(
            self.chr_seqs[sample["chrom"]],
            start_bp,
            end_bp,
            self.one_hot_table,
        )
        tad_label = self._load_targets(sample["cell"], sample["chrom"], sample["start_bp"])
        return seq, hic_target, tad_label, sample

    def __getitem__(self, index: int):
        jitter = random.randint(-200, 200) if self.mode == "train" else 0
        seq, hic_target, tad_label, sample = self.load_raw_data(index, jitter=jitter)

        if self.mode == "train" and random.random() < self.rc_prob:
            seq = reverse_complement_one_hot(seq)
            hic_target = np.flip(hic_target, axis=(0, 1)).copy()
            tad_label = np.flip(tad_label, axis=0).copy()

        if self.downsample_hic:
            if self.mode == "train":
                k = random.uniform(*self.downsample_k_range)
                hic_target = downsampling_deephic(hic_target, float(k**2))
            elif self.mode == "val":
                hic_target = downsampling_deephic(hic_target, 16.0)

        seq_tensor = torch.from_numpy(seq).float().transpose(0, 1)
        tad_label_tensor = torch.from_numpy(tad_label).long()
        return (
            seq_tensor,
            hic_to_tensor(hic_target),
            tad_label_tensor,
            sample["chrom"],
            int(sample["start_bp"]),
        )


DNA2TadMultiCellDataset = DNA2TadDataset
