import argparse
import importlib.util
import os
import sys
from pathlib import Path

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
    DEFAULT_BEDPE_DIR,
    DEFAULT_HIC_ROOT,
    DEFAULT_SEQ_ROOT,
    FULL_SPLITS,
    SMOKE_CHROMS,
    TAD_BEDPE_FILENAMES,
    default_tad_bedpe_map,
    ensure_dir,
    load_matching_state_dict,
    load_path_map,
    parse_chroms,
    resolve_device,
    seed_everything,
)
from data_processing.tad_prediction.dataset import DNA2TadDataset  # noqa: E402

def _load_local_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


try:
    from .model import DnaHicTadPredictor
except ImportError:
    DnaHicTadPredictor = _load_local_module(
        "chromseek_tad_model",
        MODULE_DIR / "model.py",
    ).DnaHicTadPredictor


DEFAULT_PRETRAINED_PATH = str(CHROMSEEK_ROOT / "checkpoints" / "chromSeek_hic_enhancement.pth")
DEFAULT_OUTPUT_DIR = str(MODULE_DIR / "runs")
DEFAULT_CELLS = list(TAD_BEDPE_FILENAMES.keys())


def compute_accuracy(preds, targets):
    pred_classes = preds.argmax(dim=1)
    return (pred_classes == targets).sum().item() / max(1, targets.numel())


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train ChromSeek TAD boundary recognition.")
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
    parser.add_argument("--tad_bedpe_dir", default=DEFAULT_BEDPE_DIR)
    parser.add_argument("--tad_bedpe_map", default="")
    parser.add_argument("--cell_lines", default=",".join(DEFAULT_CELLS))
    parser.add_argument("--train_chroms", default="")
    parser.add_argument("--val_chroms", default="")
    return parser


def make_tad_map(args):
    if args.tad_bedpe_map:
        return load_path_map(args.tad_bedpe_map)
    return default_tad_bedpe_map(args.tad_bedpe_dir)


def main():
    args = build_arg_parser().parse_args()
    seed_everything(args.seed)

    device = resolve_device(args.device)
    use_amp = device.type == "cuda"
    out_dir = ensure_dir(os.path.abspath(args.output_dir))
    log_path = os.path.join(out_dir, "train_log.txt")
    tad_bedpe_map = make_tad_map(args)

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

    ds_train = DNA2TadDataset(
        hic_root=args.hic_root,
        seq_root=args.seq_root,
        cell_lines_list=cells,
        chroms=train_chroms,
        mode="train",
        rc_prob=0.5,
        max_samples=max_samples,
        downsample_hic=True,
        tad_bedpe_map=tad_bedpe_map,
    )
    ds_val = DNA2TadDataset(
        hic_root=args.hic_root,
        seq_root=args.seq_root,
        cell_lines_list=cells,
        chroms=val_chroms,
        mode="val",
        rc_prob=0.0,
        max_samples=max_samples,
        downsample_hic=True,
        tad_bedpe_map=tad_bedpe_map,
    )
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_amp)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_amp)

    model = DnaHicTadPredictor(pretrained_path=None)
    load_matching_state_dict(model, args.pretrained_path, label="enhanced Hi-C transfer checkpoint")
    model = model.to(device)

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
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 30.0], device=device))
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        steps = 0
        for batch in tqdm(dl_train, desc=f"Ep{epoch} [Train]"):
            seq, hic, target = batch[:3]
            seq, hic, target = seq.to(device), hic.to(device), target.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                pred = model(seq, hic)
                loss = loss_fn(pred, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            train_acc += compute_accuracy(pred.detach(), target)
            steps += 1

        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for batch in tqdm(dl_val, desc=f"Ep{epoch} [Val]"):
                seq, hic, target = batch[:3]
                seq, hic, target = seq.to(device), hic.to(device), target.to(device)
                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                    pred = model(seq, hic)
                    loss = loss_fn(pred, target)
                val_loss += loss.item()
                val_acc += compute_accuracy(pred, target)

        avg_train_loss = train_loss / max(1, steps)
        avg_train_acc = train_acc / max(1, steps)
        avg_val_loss = val_loss / max(1, len(dl_val))
        avg_val_acc = val_acc / max(1, len(dl_val))
        line = (
            f"Epoch {epoch}: TrainLoss {avg_train_loss:.4f}, TrainAcc {avg_train_acc:.4f}, "
            f"ValLoss {avg_val_loss:.4f}, ValAcc {avg_val_acc:.4f}"
        )
        print(line)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(out_dir, "chromSeek_tad_prediction_best.pth"))
            print(f"Saved best model (ValLoss: {best_val_loss:.4f})")


if __name__ == "__main__":
    main()
