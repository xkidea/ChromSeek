import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "downstreams" / "tad_prediction" / "sample_data.pt"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_processing.tad_prediction.dataset import (  # noqa: E402
    DNA2TadMultiCellDataset,
)


def main():
    val_chroms = ["chr18", "chr19"]
    cell_lines_list = ["GM12878_Hic"]
    HIC_ROOT = "/mnt/nfs/jyzhu/dataset/lets_fk_hic/hic_patches_10kb_multiratio/"
    SEQ_ROOT = "/mnt/nfs/jyzhu/proj/ChromSeek/DNA_ChromSeek/DNA_only_10kb/genome_cache/"
    ds = DNA2TadMultiCellDataset(
        hic_root=HIC_ROOT,
        seq_root=SEQ_ROOT,
        cell_lines_list=cell_lines_list,
        chroms_list=val_chroms,
        mode="val",
        augment_rc=False,
    )
    idx_to_save = -1
    for i in range(len(ds)):
        _, raw_hic, tad_label, _ = ds.load_raw_data(i, jitter=0)
        num_tads = np.sum(tad_label)
        if 5 <= num_tads <= 12:
            idx_to_save = i
            break
    if idx_to_save != -1:
        res = ds[idx_to_save]
        t_seq, t_hic_in, t_tad_target = res[0], res[1], res[2]
        _, raw_hic, _, _ = ds.load_raw_data(idx_to_save, jitter=0)
        sample_out = {
            "inputs": {"seq": t_seq, "hic": t_hic_in},
            "targets": {"tad": t_tad_target},
            "info": {"raw_hic": raw_hic},
        }
        torch.save([sample_out], OUTPUT_PATH)
        print(f"Extracted real TAD sample at index {idx_to_save} with {np.sum(t_tad_target.numpy())} boundaries.")
    else:
        print("Failed to find a sample.")
if __name__ == "__main__":
    main()
