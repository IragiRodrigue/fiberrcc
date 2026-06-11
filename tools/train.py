#!/usr/bin/env python3
"""
train.py
=========
Entry point for FiberRCNN training.

Single-GPU:
    python tools/train.py \\
        --config configs/fiber_rcnn_r50_fpn.yaml \\
        --train_json  /data/coco_fiber/train.json \\
        --val_json    /data/coco_fiber/val.json \\
        --image_root  /data/images \\
        --output_dir  ./output/run01

Multi-GPU (DDP, 4 GPUs):
    python -m torch.distributed.run --nproc_per_node=4 tools/train.py \\
        --config configs/fiber_rcnn_r50_fpn.yaml \\
        --train_json /data/coco_fiber/train.json \\
        --val_json   /data/coco_fiber/val.json \\
        --image_root /data/images \\
        --output_dir ./output/run01 \\
        --num_gpus   4

Resume from checkpoint:
    python tools/train.py ... --resume
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from detectron2.utils import comm
from detectron2.utils.logger import setup_logger
from loguru import logger as loguru_logger


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train FiberRCNN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=Path, required=True, help="YAML config file.")
    p.add_argument("--train_json", type=Path, required=True, help="COCO-Fiber train JSON.")
    p.add_argument("--val_json", type=Path, required=True, help="COCO-Fiber val JSON.")
    p.add_argument("--image_root", type=Path, required=True, help="Image directory.")
    p.add_argument("--output_dir", type=Path, required=True, help="Output directory.")
    p.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs.")
    p.add_argument("--resume", action="store_true", help="Resume from last checkpoint.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--wandb_project", type=str, default=None, help="W&B project name.")
    p.add_argument("--wandb_run", type=str, default="fiberrcnn", help="W&B run name.")
    p.add_argument(
        "--early_stop_patience",
        type=int,
        default=20,
        help="Early stopping patience (0 = disabled).",
    )
    p.add_argument(
        "--opts",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional Detectron2 config overrides (KEY VALUE ...).",
    )
    return p.parse_args()


def _setup_logging(output_dir: Path) -> None:
    setup_logger(output_dir, distributed_rank=comm.get_rank(), name="detectron2")
    loguru_logger.remove()
    loguru_logger.add(sys.stderr, level="INFO")
    if comm.is_main_process():
        loguru_logger.add(output_dir / "train.log", level="DEBUG", rotation="50 MB")


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(args.output_dir)

    # ---- Imports after logging setup ----
    import fiberrcnn  # triggers all registrations
    from fiberrcnn.data import register_coco_fiber_dataset
    from fiberrcnn.engine.trainer import FiberTrainer, build_fiber_cfg

    # Register datasets
    register_coco_fiber_dataset("fiber_train", args.train_json, args.image_root)
    register_coco_fiber_dataset("fiber_val", args.val_json, args.image_root)

    # Build config
    overrides: dict = {}
    if args.opts:
        it = iter(args.opts)
        for k in it:
            overrides[k] = next(it)

    cfg = build_fiber_cfg(
        base_config_file=str(args.config),
        dataset_train="fiber_train",
        dataset_val="fiber_val",
        output_dir=str(args.output_dir),
        overrides=overrides,
    )

    loguru_logger.info(f"Config:\n{cfg}")

    trainer = FiberTrainer(
        cfg,
        seed=args.seed,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run,
        early_stop_patience=args.early_stop_patience,
    )

    trainer.resume_or_load(resume=args.resume)

    try:
        trainer.train()
    except StopIteration:
        loguru_logger.info("Training stopped early.")

    loguru_logger.success(f"Training complete. Output → {args.output_dir}")


if __name__ == "__main__":
    main()
