import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from tqdm import tqdm

MODULE_DIR = Path(__file__).resolve().parent
CHROMSEEK_ROOT = MODULE_DIR.parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))
if str(CHROMSEEK_ROOT) not in sys.path:
    sys.path.insert(0, str(CHROMSEEK_ROOT))

from data_processing.common import (  # noqa: E402
    DEFAULT_HIC_ROOT,
    DEFAULT_SEQ_ROOT,
    FULL_SPLITS,
    SMOKE_CHROMS,
    ensure_dir,
    parse_chroms,
    resolve_device,
    seed_everything,
)
from data_processing.hic_enhancement.dataset import GeneralHiCDataset  # noqa: E402

def _load_local_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


try:
    from .model import StrongFusionModel_V4
except ImportError:
    StrongFusionModel_V4 = _load_local_module(
        "chromseek_hic_enhancement_model",
        MODULE_DIR / "model.py",
    ).StrongFusionModel_V4


DEFAULT_PRETRAINED_PATH = str(CHROMSEEK_ROOT / "checkpoints" / "best_model_448k_200bp.pth")
DEFAULT_OUTPUT_DIR = str(MODULE_DIR / "runs")
DEFAULT_CELLS = [
    "A673_Hic",
    "Caco2_Hic",
    "Calu3_Hic",
    "GM12878_Hic",
    "GM23248_Hic",
    "Hct116_Hic",
    "HepG2_Hic",
    "K562_Hic",
    "MCF10A_Hic",
    "Mcf7_Hic",
    "OCILY7_Hic",
    "PC3_Hic",
    "PC9_Hic",
]


def pearson_corr_loss(pred, target):
    pred = pred.view(-1, pred.shape[-1])
    target = target.view(-1, target.shape[-1])
    pred_c = pred - pred.mean(dim=1, keepdim=True)
    target_c = target - target.mean(dim=1, keepdim=True)
    cov = (pred_c * target_c).sum(dim=1)
    pred_var = (pred_c**2).sum(dim=1) + 1e-8
    target_var = (target_c**2).sum(dim=1) + 1e-8
    corr = cov / torch.sqrt(pred_var * target_var)
    return 1.0 - corr.mean()


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train ChromSeek Hi-C enhancement.")
    parser.add_argument("--hic_root", default=DEFAULT_HIC_ROOT)
    parser.add_argument("--seq_root", default=DEFAULT_SEQ_ROOT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--pretrained_path", default=DEFAULT_PRETRAINED_PATH)
    parser.add_argument("--cell_lines", default=",".join(DEFAULT_CELLS))
    parser.add_argument("--train_chroms", default="")
    parser.add_argument("--val_chroms", default="")
    return parser


def main():
    args = build_arg_parser().parse_args()
    seed_everything(args.seed)

    device = resolve_device(args.device)
    use_amp = device.type == "cuda"
    out_dir = ensure_dir(os.path.abspath(args.output_dir))
    log_path = os.path.join(out_dir, "train_log.txt")

    if args.smoke_test or args.dry_run:
        epochs = min(args.epochs, 2)
        cells = ["GM12878_Hic"]
        train_chroms = parse_chroms(args.train_chroms) or SMOKE_CHROMS
        val_chroms = parse_chroms(args.val_chroms) or SMOKE_CHROMS
        max_samples = 8
        num_workers = 0
    else:
        epochs = args.epochs
        cells = parse_chroms(args.cell_lines)
        train_chroms = parse_chroms(args.train_chroms) or FULL_SPLITS["train"]
        val_chroms = parse_chroms(args.val_chroms) or FULL_SPLITS["val"]
        max_samples = None
        num_workers = args.num_workers

    ds_train = GeneralHiCDataset(args.hic_root, args.seq_root, cells, train_chroms, "train", 0.5, max_samples)
    ds_val = GeneralHiCDataset(args.hic_root, args.seq_root, cells, val_chroms, "val", 0.0, max_samples)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_amp)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_amp)

    model = StrongFusionModel_V4(pretrained_path=args.pretrained_path if args.pretrained_path else None).to(device)
    if use_amp and torch.cuda.device_count() > 1 and not (args.smoke_test or args.dry_run):
        model = nn.DataParallel(model)

    print(f"Using device: {device}")
    print(f"Train Dataset size: {len(ds_train)}")
    print(f"Val Dataset size: {len(ds_val)}")
    print(f"Output directory: {out_dir}")
    if args.dry_run:
        print("Dry run complete: dataset and model initialization succeeded.")
        return
    if len(ds_train) == 0 or len(ds_val) == 0:
        raise RuntimeError("Training and validation datasets must both be non-empty.")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    mse_loss_fn = nn.MSELoss()
    best_pcc = -1.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        steps = 0
        for seq, hic_in, hic_gt in tqdm(dl_train, desc=f"Ep{epoch} [Train]"):
            seq, hic_in, hic_gt = seq.to(device), hic_in.to(device), hic_gt.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                _, pred_2d = model(seq, hic_in)
                pred_2d = pred_2d.squeeze(1)
                loss = mse_loss_fn(pred_2d, hic_gt) + 0.1 * pearson_corr_loss(pred_2d, hic_gt)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            steps += 1

        model.eval()
        val_loss = 0.0
        pcc_values = []
        with torch.no_grad():
            for seq, hic_in, hic_gt in tqdm(dl_val, desc=f"Ep{epoch} [Val]"):
                seq, hic_in, hic_gt = seq.to(device), hic_in.to(device), hic_gt.to(device)
                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                    _, pred_2d = model(seq, hic_in)
                    pred_2d = pred_2d.squeeze(1)
                    loss = mse_loss_fn(pred_2d, hic_gt) + 0.1 * pearson_corr_loss(pred_2d, hic_gt)
                val_loss += loss.item()
                pred_np, gt_np = pred_2d.cpu().numpy(), hic_gt.cpu().numpy()
                for batch_idx in range(pred_np.shape[0]):
                    try:
                        pcc, _ = pearsonr(pred_np[batch_idx].ravel(), gt_np[batch_idx].ravel())
                        if not np.isnan(pcc):
                            pcc_values.append(pcc)
                    except ValueError:
                        pass

        avg_train_loss = train_loss / max(1, steps)
        avg_val_loss = val_loss / max(1, len(dl_val))
        avg_pcc = float(np.mean(pcc_values)) if pcc_values else 0.0
        line = (
            f"Epoch {epoch}: TrainLoss {avg_train_loss:.4f}, ValLoss {avg_val_loss:.4f}, "
            f"ValPCC {avg_pcc:.4f}, Time {time.time() - t0:.1f}s"
        )
        print(line)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

        if avg_pcc > best_pcc:
            best_pcc = avg_pcc
            state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save(state, os.path.join(out_dir, "chromSeek_hic_enhancement_best.pth"))
            print(f"Saved best model (PCC: {best_pcc:.4f})")


if __name__ == "__main__":
    main()
