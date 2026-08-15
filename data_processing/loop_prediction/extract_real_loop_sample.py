import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "downstreams" / "loop_prediction" / "sample_data.pt"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_processing.loop_prediction.dataset import DNA2LoopDataset  # noqa: E402


def main():
    ds = DNA2LoopDataset(
        hic_root="/mnt/nfs/jyzhu/dataset/lets_fk_hic/hic_patches_10kb_multiratio/",
        seq_root="/mnt/nfs/jyzhu/proj/ChromSeek/DNA_ChromSeek/DNA_only_10kb/genome_cache",
        loop_paths=["/mnt/nfs/jyzhu/dataset/lets_fk_hic/ENCODE_Data/Gm12878_Loop.bedpe.gz"],
        chroms=["chr18"],
        mode="val",
        rc_prob=0.0,
        max_samples=None,
        downsample_hic=False,
    )
    idx_to_save = -1
    for i in range(len(ds)):
        _, raw_hic, target_loop, _ = ds.load_raw_data(i, jitter=0)
        num_loops = np.sum(target_loop)
        if 2 <= num_loops <= 10:
            idx_to_save = i
            break
    if idx_to_save != -1:
        res = ds[idx_to_save]
        t_seq, t_hic_in, t_loop_target = res[0], res[1], res[2]
        _, raw_hic, _, _ = ds.load_raw_data(idx_to_save, jitter=0)
        sample_out = {
            "inputs": {"seq": t_seq, "hic": t_hic_in},
            "targets": {"loop": t_loop_target},
            "info": {"raw_hic": raw_hic},
        }
        torch.save([sample_out], OUTPUT_PATH)
        print(f"Extracted real Loop sample at index {idx_to_save} with {np.sum(target_loop)} loop anchors.")
    else:
        print("Failed to find a sample.")
if __name__ == "__main__":
    main()
