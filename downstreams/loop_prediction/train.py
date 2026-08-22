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
    default_loop_paths,
    ensure_dir,
    load_matching_state_dict,
    parse_chroms,
    resolve_device,
    seed_everything,
)
from data_processing.loop_prediction.dataset import DNA2LoopDataset  # noqa: E402

def _load_local_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


try:
    from .model import LoopPredictionModel
except ImportError:
    LoopPredictionModel = _load_local_module(
        "chromseek_loop_model",
        MODULE_DIR / "model.py",
    ).LoopPredictionModel


DEFAULT_PRETRAINED_PATH = str(CHROMSEEK_ROOT / "checkpoints" / "chromSeek_hic_enhancement.pth")
DEFAULT_OUTPUT_DIR = str(MODULE_DIR / "runs")


def compute_metrics(pred_logits, target_labels):
    pred_classes = pred_logits.argmax(dim=1)
    tp = (pred_classes * target_labels).sum().float()
    fp = (pred_classes * (1 - target_labels)).sum().float()
    fn = ((1 - pred_classes) * target_labels).sum().float()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision.item(), recall.item(), f1.item()


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train ChromSeek loop recognition.")
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
    parser.add_argument("--loop_bedpe_dir", default=DEFAULT_BEDPE_DIR)
    parser.add_argument("--loop_paths", nargs="*", default=None)
    parser.add_argument("--train_chroms", default="")
    parser.add_argument("--val_chroms", default="")
    return parser


def resolve_loop_paths(args):
    loop_paths = args.loop_paths if args.loop_paths else default_loop_paths(args.loop_bedpe_dir)
    if args.smoke_test or args.dry_run:
        gm12878 = [path for path in loop_paths if os.path.basename(path).lower().startswith("gm12878")]
        return gm12878 or loop_paths[:1]
    return loop_paths


def main():
    args = build_arg_parser().parse_args()
    seed_everything(args.seed)

    device = resolve_device(args.device)
    use_amp = device.type == "cuda"
    out_dir = ensure_dir(os.path.abspath(args.output_dir))
    log_path = os.path.join(out_dir, "train_log.txt")
    loop_paths = resolve_loop_paths(args)

    if args.smoke_test or args.dry_run:
        epochs = min(args.epochs, 2)
        train_chroms = parse_chroms(args.train_chroms) or SMOKE_CHROMS
        val_chroms = parse_chroms(args.val_chroms) or SMOKE_CHROMS
        max_samples = 8
        num_workers = 0
    else:
        epochs = args.epochs
        train_chroms = parse_chroms(args.train_chroms) or FULL_SPLITS["train"]
        val_chroms = parse_chroms(args.val_chroms) or FULL_SPLITS["val"]
        max_samples = None
        num_workers = args.num_workers

    ds_train = DNA2LoopDataset(
        hic_root=args.hic_root,
        seq_root=args.seq_root,
        loop_paths=loop_paths,
        chroms=train_chroms,
        mode="train",
        rc_prob=0.5,
        downsample_hic=True,
        max_samples=max_samples,
    )
    ds_val = DNA2LoopDataset(
        hic_root=args.hic_root,
        seq_root=args.seq_root,
        loop_paths=loop_paths,
        chroms=val_chroms,
        mode="val",
        rc_prob=0.0,
        downsample_hic=True,
        max_samples=max_samples,
    )
    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=not (args.smoke_test or args.dry_run),
        pin_memory=use_amp,
    )
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_amp)

    model = LoopPredictionModel(num_tasks_2d=2, pretrained_path=None)
    load_matching_state_dict(model, args.pretrained_path, label="enhanced Hi-C transfer checkpoint")
    model = model.to(device)

    print(f"Using device: {device}")
    print(f"Train Dataset size: {len(ds_train)}")
    print(f"Val Dataset size: {len(ds_val)}")
    print(f"Loop BEDPE files: {len(loop_paths)}")
    print(f"Output directory: {out_dir}")
    if args.dry_run:
        print("Dry run complete: dataset and model initialization succeeded.")
        return
    if len(ds_train) == 0 or len(ds_val) == 0:
        raise RuntimeError("Training and validation datasets must both be non-empty.")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 100.0], device=device))
    best_val_f1 = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        steps = 0
        for batch in tqdm(dl_train, desc=f"Ep{epoch} [Train]"):
            seq, hic, target_loop = batch[:3]
            seq, hic, target_loop = seq.to(device), hic.to(device), target_loop.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                pred_logits = model(seq, hic)
                loss = criterion(pred_logits, target_loop)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            steps += 1

        model.eval()
        val_loss = 0.0
        val_prec = 0.0
        val_rec = 0.0
        val_f1 = 0.0
        with torch.no_grad():
            for batch in tqdm(dl_val, desc=f"Ep{epoch} [Val]"):
                seq, hic, target_loop = batch[:3]
                seq, hic, target_loop = seq.to(device), hic.to(device), target_loop.to(device)
                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                    pred_logits = model(seq, hic)
                    loss = criterion(pred_logits, target_loop)
                val_loss += loss.item()
                precision, recall, f1 = compute_metrics(pred_logits, target_loop)
                val_prec += precision
                val_rec += recall
                val_f1 += f1

        avg_train_loss = train_loss / max(1, steps)
        avg_val_loss = val_loss / max(1, len(dl_val))
        avg_val_prec = val_prec / max(1, len(dl_val))
        avg_val_rec = val_rec / max(1, len(dl_val))
        avg_val_f1 = val_f1 / max(1, len(dl_val))
        line = (
            f"Epoch {epoch}: TrainLoss {avg_train_loss:.4f}, ValLoss {avg_val_loss:.4f}, "
            f"Precision {avg_val_prec:.4f}, Recall {avg_val_rec:.4f}, F1 {avg_val_f1:.4f}"
        )
        print(line)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

        if avg_val_f1 > best_val_f1:
            best_val_f1 = avg_val_f1
            torch.save(model.state_dict(), os.path.join(out_dir, "chromSeek_loop_prediction_best.pth"))
            print(f"Saved best model (Val F1: {best_val_f1:.4f})")


if __name__ == "__main__":
    main()
