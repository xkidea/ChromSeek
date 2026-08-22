import os
import random
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from data_processing.common import (
    BIN_SIZE,
    DEFAULT_MULTIOMICS_PREPROCESSED_DIR,
    DNA_BP_WINDOW,
    HIC_RESOLUTION,
    MULTIOMICS_TRACK_NAMES,
    NUM_1KB_BINS,
    downsampling_deephic,
    encode_sequence_window,
    hic_to_tensor,
    load_json,
    reverse_complement_one_hot,
)
from data_processing.dna_loader import build_one_hot_table, load_chr


class DNA2MultiOmicsDataset(Dataset):
    def __init__(
        self,
        hic_roots: Dict[str, str],
        seq_root: str,
        preprocessed_dir: str = DEFAULT_MULTIOMICS_PREPROCESSED_DIR,
        chroms: Sequence[str] = (),
        mode: str = "train",
        rc_prob: float = 0.5,
        max_samples: Optional[int] = None,
        sample_stride: int = 1,
    ) -> None:
        super().__init__()
        self.hic_roots = dict(hic_roots)
        self.cells = list(hic_roots.keys())
        self.seq_root = seq_root
        self.preprocessed_dir = preprocessed_dir
        self.chroms = list(chroms)
        self.mode = mode
        self.rc_prob = rc_prob
        self.max_samples = max_samples
        self.sample_stride = max(1, sample_stride)

        self.one_hot_table = build_one_hot_table()
        self.stats = load_json(os.path.join(preprocessed_dir, "stats", "train_stats.json"))
        self.track_order = self.stats.get("track_order", MULTIOMICS_TRACK_NAMES)

        self.track_arrays: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {cell: {} for cell in self.cells}
        for cell in self.cells:
            track_dir = os.path.join(preprocessed_dir, f"tracks_{cell}")
            for track in self.track_order:
                track_path = os.path.join(track_dir, f"{track}.npz")
                with np.load(track_path, allow_pickle=False) as handle:
                    self.track_arrays[cell][track] = {
                        chrom: handle[chrom] for chrom in handle.files if chrom in self.chroms
                    }

        self.chr_seqs = {chrom: load_chr(chrom, seq_root, mmap=True) for chrom in self.chroms}
        self.samples = self._collect_samples()
        if self.max_samples is not None:
            self.samples = self.samples[: self.max_samples]

    def _collect_samples(self):
        samples = []
        for cell in self.cells:
            hic_root = self.hic_roots[cell]
            for chrom in self.chroms:
                chrom_dir = os.path.join(hic_root, chrom)
                meta_path = os.path.join(chrom_dir, "meta.txt")
                if not os.path.exists(meta_path):
                    continue
                with open(meta_path, "r", encoding="utf-8") as handle:
                    starts = [int(line.strip()) for line in handle if line.strip()]

                for idx, start_idx in enumerate(starts):
                    if idx % self.sample_stride != 0:
                        continue
                    start_bp = start_idx * HIC_RESOLUTION
                    end_bp = start_bp + DNA_BP_WINDOW
                    start_bin = start_bp // BIN_SIZE
                    end_bin = start_bin + NUM_1KB_BINS
                    if chrom not in self.chr_seqs or end_bp > len(self.chr_seqs[chrom]):
                        continue
                    if any(chrom not in self.track_arrays[cell][track] for track in self.track_order):
                        continue
                    if any(end_bin > len(self.track_arrays[cell][track][chrom]) for track in self.track_order):
                        continue
                    pt_path = os.path.join(chrom_dir, f"{start_idx}.pt")
                    if not os.path.exists(pt_path):
                        continue
                    samples.append(
                        {
                            "cell": cell,
                            "chrom": chrom,
                            "start_idx": start_idx,
                            "start_bp": start_bp,
                            "pt_path": pt_path,
                        }
                    )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_hic_high_depth(self, pt_path: str):
        hic_data = torch.load(pt_path, map_location="cpu", weights_only=False)
        if isinstance(hic_data, dict) and "target" in hic_data:
            hic_data = hic_data["target"]
        if hasattr(hic_data, "numpy"):
            hic_data = hic_data.numpy()
        hic_target = np.asarray(hic_data, dtype=np.float32)
        hic_target = np.nan_to_num(hic_target, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        return np.clip(hic_target, a_min=0.0, a_max=None)

    def _load_targets(self, cell: str, chrom: str, start_bp: int):
        start_bin = start_bp // BIN_SIZE
        end_bin = start_bin + NUM_1KB_BINS
        raw_tracks = []
        z_tracks = []
        for track in self.track_order:
            values = self.track_arrays[cell][track][chrom][start_bin:end_bin].astype(np.float32, copy=False)
            values_log = np.log1p(values)
            stats = self.stats["tracks"][track]
            values_z = (values_log - stats["mean"]) / stats["std"]
            raw_tracks.append(values_log)
            z_tracks.append(values_z)
        return np.stack(z_tracks, axis=0), np.stack(raw_tracks, axis=0)

    def load_raw_data(self, index: int):
        sample = self.samples[index]
        seq = encode_sequence_window(
            self.chr_seqs[sample["chrom"]],
            sample["start_bp"],
            sample["start_bp"] + DNA_BP_WINDOW,
            self.one_hot_table,
        )
        hic_target = self._load_hic_high_depth(sample["pt_path"])
        target_z, target_log = self._load_targets(sample["cell"], sample["chrom"], sample["start_bp"])
        return seq, hic_target, target_z, target_log, sample

    def __getitem__(self, index: int):
        seq, hic_target, target_z, target_log, sample = self.load_raw_data(index)

        if self.mode == "train" and random.random() < self.rc_prob:
            seq = reverse_complement_one_hot(seq)
            hic_target = np.flip(hic_target, axis=(0, 1)).copy()
            target_z = np.flip(target_z, axis=1).copy()
            target_log = np.flip(target_log, axis=1).copy()

        if self.mode == "train":
            k = random.uniform(2, 25)
            hic_target = downsampling_deephic(hic_target, float(k**2))
        else:
            hic_target = downsampling_deephic(hic_target, 16.0)

        return (
            torch.from_numpy(seq).float().transpose(0, 1),
            hic_to_tensor(hic_target),
            torch.from_numpy(target_z).float(),
            torch.from_numpy(target_log).float(),
            sample["chrom"],
            int(sample["start_bp"]),
            sample["cell"],
        )
