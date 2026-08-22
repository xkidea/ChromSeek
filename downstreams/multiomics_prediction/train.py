import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

MODULE_DIR = Path(__file__).resolve().parent
CHROMSEEK_ROOT = MODULE_DIR.parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))
if str(CHROMSEEK_ROOT) not in sys.path:
    sys.path.insert(0, str(CHROMSEEK_ROOT))

from data_processing.common import (  # noqa: E402
    DEFAULT_HIC_ROOT,
    DEFAULT_MULTIOMICS_HIC_ROOTS,
    DEFAULT_MULTIOMICS_PREPROCESSED_DIR,
    DEFAULT_SEQ_ROOT,
    FULL_SPLITS,
    SMOKE_CHROMS,
    ensure_dir,
    load_matching_state_dict,
    load_path_map,
    parse_chroms,
    pearson_corr_loss,
    pearson_corr_np,
    resolve_device,
    seed_everything,
)
from data_processing.multiomics_prediction.dataset import (  # noqa: E402
    DNA2MultiOmicsDataset,
)


def _load_local_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


try:
    from .multitask_model import MultiOmicsPredictor
except ImportError:
    MultiOmicsPredictor = _load_local_module(
        "chromseek_multiomics_multitask_model",
        MODULE_DIR / "multitask_model.py",
    ).MultiOmicsPredictor


DEFAULT_PRETRAINED_PATH = str(CHROMSEEK_ROOT / "checkpoints" / "chromSeek_hic_enhancement.pth")
DEFAULT_OUTPUT_DIR = str(MODULE_DIR / "runs")
DEFAULT_CELLS = ["GM12878", "K562"]


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train ChromSeek multi-omics prediction.")
    parser.add_argument("--hic_root", default=DEFAULT_HIC_ROOT)
    parser.add_argument("--hic_roots_map", default="", help="Optional JSON/TSV map from cell name to Hi-C patch dir.")
    parser.add_argument("--seq_root", default=DEFAULT_SEQ_ROOT)
    parser.add_argument("--preprocessed_dir", default=DEFAULT_MULTIOMICS_PREPROCESSED_DIR)
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
    parser.add_argument("--cells", default=",".join(DEFAULT_CELLS))
    parser.add_argument("--train_chroms", default="")
    parser.add_argument("--val_chroms", default="")
    parser.add_argument("--max_samples", type=int, default=0)
    return parser


def resolve_hic_roots(args, cells):
    if args.hic_roots_map:
        mapping = load_path_map(args.hic_roots_map)
        return {cell: mapping[cell] for cell in cells if cell in mapping}

    roots = {}
    for cell in cells:
        cell_dir = os.path.join(args.hic_root, f"{cell}_Hic")
        if os.path.isdir(cell_dir):
            roots[cell] = cell_dir
        elif cell in DEFAULT_MULTIOMICS_HIC_ROOTS:
            roots[cell] = DEFAULT_MULTIOMICS_HIC_ROOTS[cell]
        elif len(cells) == 1 and os.path.isdir(args.hic_root):
            roots[cell] = args.hic_root
    return roots


def main():
    args = build_arg_parser().parse_args()
    seed_everything(args.seed)

    device = resolve_device(args.device)
    use_amp = device.type == "cuda"
    out_dir = ensure_dir(os.path.abspath(args.output_dir))
    log_path = os.path.join(out_dir, "train_log.txt")

    if args.smoke_test or args.dry_run:
        epochs = min(args.epochs, 2)
        cells = ["GM12878"]
        train_chroms = parse_chroms(args.train_chroms) or SMOKE_CHROMS
        val_chroms = parse_chroms(args.val_chroms) or SMOKE_CHROMS
        max_samples = args.max_samples or 8
        num_workers = 0
    else:
        epochs = args.epochs
        cells = parse_chroms(args.cells)
        train_chroms = parse_chroms(args.train_chroms) or FULL_SPLITS["train"]
        val_chroms = parse_chroms(args.val_chroms) or FULL_SPLITS["val"]
        max_samples = args.max_samples or None
        num_workers = args.num_workers

    hic_roots = resolve_hic_roots(args, cells)
    missing_cells = [cell for cell in cells if cell not in hic_roots]
    if missing_cells:
        raise FileNotFoundError(f"Missing Hi-C roots for cells: {missing_cells}")

    ds_train = DNA2MultiOmicsDataset(
        hic_roots=hic_roots,
        seq_root=args.seq_root,
        preprocessed_dir=args.preprocessed_dir,
        chroms=train_chroms,
        mode="train",
        rc_prob=0.5,
        max_samples=max_samples,
    )
    ds_val = DNA2MultiOmicsDataset(
        hic_roots=hic_roots,
        seq_root=args.seq_root,
        preprocessed_dir=args.preprocessed_dir,
        chroms=val_chroms,
        mode="val",
        rc_prob=0.0,
        max_samples=max_samples,
    )
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_amp)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_amp)

    model = MultiOmicsPredictor(track_names=ds_train.track_order, cells=cells, pretrained_path=None)
    load_matching_state_dict(model, args.pretrained_path, label="enhanced Hi-C transfer checkpoint")
    model = model.to(device)

    print(f"Using device: {device}")
    print(f"Cells: {','.join(cells)}")
    print(f"Tracks: {len(ds_train.track_order)}")
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
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        steps = 0
        for seq, hic, target_z, _, _, _, cells_batch in tqdm(dl_train, desc=f"Ep{epoch} [Train]"):
            seq = seq.to(device)
            hic = hic.to(device)
            target_z = target_z.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                pred_z = model(seq, hic, list(cells_batch))
                loss = mse_loss_fn(pred_z, target_z) + 0.1 * pearson_corr_loss(pred_z, target_z)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            steps += 1

        model.eval()
        val_loss = 0.0
        pcc_values = []
        with torch.no_grad():
            for seq, hic, target_z, _, _, _, cells_batch in tqdm(dl_val, desc=f"Ep{epoch} [Val]"):
                seq = seq.to(device)
                hic = hic.to(device)
                target_z = target_z.to(device)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    pred_z = model(seq, hic, list(cells_batch))
                    loss = mse_loss_fn(pred_z, target_z) + 0.1 * pearson_corr_loss(pred_z, target_z)
                val_loss += loss.item()

                pred_np = pred_z.cpu().numpy()
                true_np = target_z.cpu().numpy()
                for batch_idx in range(pred_np.shape[0]):
                    pcc = pearson_corr_np(pred_np[batch_idx], true_np[batch_idx])
                    if not np.isnan(pcc):
                        pcc_values.append(pcc)

        avg_train_loss = train_loss / max(1, steps)
        avg_val_loss = val_loss / max(1, len(dl_val))
        avg_pcc = float(np.mean(pcc_values)) if pcc_values else 0.0
        line = (
            f"Epoch {epoch}: TrainLoss {avg_train_loss:.4f}, "
            f"ValLoss {avg_val_loss:.4f}, ValPCC {avg_pcc:.4f}"
        )
        print(line)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(out_dir, "chromSeek_multiomics_prediction_best.pth"))
            print(f"Saved best model (ValLoss: {best_val_loss:.4f})")


if __name__ == "__main__":
    main()
