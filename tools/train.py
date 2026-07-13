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

Resume from checkpoint:
    python tools/train.py ... --resume
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train FiberRCNN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",      type=Path, required=True)
    p.add_argument("--train_json",  type=Path, required=True)
    p.add_argument("--val_json",    type=Path, required=True)
    p.add_argument("--image_root",  type=Path, required=True)
    p.add_argument("--output_dir",  type=Path, required=True)
    p.add_argument("--resume",      action="store_true")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_run",   type=str, default="fiberrcnn")
    p.add_argument("--early_stop_patience", type=int, default=20)
    p.add_argument(
        "--opts",
        nargs=argparse.REMAINDER,
        default=[],
        help="Detectron2 config overrides: KEY VALUE ...",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # ── Convert all Paths to str immediately (Windows compatibility) ──
    config_file  = str(args.config)
    train_json   = str(args.train_json)
    val_json     = str(args.val_json)
    image_root   = str(args.image_root)
    output_dir   = str(args.output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # ── Logging ───────────────────────────────────────────────────────
    from detectron2.utils import comm
    from detectron2.utils.logger import setup_logger
    from loguru import logger as loguru_logger

    setup_logger(output_dir, distributed_rank=comm.get_rank(), name="detectron2")
    loguru_logger.remove()
    loguru_logger.add(sys.stderr, level="INFO")
    if comm.is_main_process():
        loguru_logger.add(
            os.path.join(output_dir, "train.log"),
            level="DEBUG",
            rotation="50 MB",
        )

    # ── Imports ───────────────────────────────────────────────────────
    import fiberrcnn  # triggers all Detectron2 registrations
    from fiberrcnn.data import register_coco_fiber_dataset
    from fiberrcnn.engine.trainer import FiberTrainer, build_fiber_cfg

    register_coco_fiber_dataset("fiber_train", train_json, image_root)
    register_coco_fiber_dataset("fiber_val",   val_json,   image_root)

    # ── Config overrides ──────────────────────────────────────────────
    overrides: dict = {}
    if args.opts:
        it = iter(args.opts)
        for k in it:
            overrides[k] = next(it)

    cfg = build_fiber_cfg(
        base_config_file=config_file,
        dataset_train="fiber_train",
        dataset_val="fiber_val",
        output_dir=output_dir,
        overrides=overrides,
    )

    loguru_logger.info(f"Output dir : {output_dir}")
    loguru_logger.info(f"Batch size : {cfg.SOLVER.IMS_PER_BATCH}")
    loguru_logger.info(f"Max iter   : {cfg.SOLVER.MAX_ITER}")

    # ── Trainer ───────────────────────────────────────────────────────
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

    loguru_logger.success(f"Training complete → {output_dir}")


if __name__ == "__main__":
    main()