"""
Logging utilities for FiberRCNN.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from loguru import logger


def setup_loguru(
    output_dir: str | Path | None = None,
    level: str = "INFO",
    rotation: str = "50 MB",
) -> None:
    """Configure loguru for the current process.

    Parameters
    ----------
    output_dir : optional directory — a ``fiberrcnn.log`` file is created there
    level : minimum log level for stderr
    rotation : loguru rotation policy for the file handler
    """
    logger.remove()
    logger.add(sys.stderr, level=level, colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        logger.add(
            Path(output_dir) / "fiberrcnn.log",
            level="DEBUG",
            rotation=rotation,
            encoding="utf-8",
        )


class TensorBoardLogger:
    """Lightweight TensorBoard wrapper.

    Parameters
    ----------
    log_dir : directory for TensorBoard event files
    """

    def __init__(self, log_dir: str | Path) -> None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(str(log_dir))
            logger.info(f"TensorBoard writer initialised → {log_dir}")
        except ImportError:
            self._writer = None
            logger.warning("tensorboard not installed — TB logging disabled.")

    def scalar(self, tag: str, value: float, step: int) -> None:
        if self._writer is not None:
            self._writer.add_scalar(tag, value, step)

    def scalars(self, tag: str, values: dict[str, float], step: int) -> None:
        if self._writer is not None:
            self._writer.add_scalars(tag, values, step)

    def image(self, tag: str, image, step: int) -> None:
        """image: (H, W, C) uint8 numpy or (C, H, W) float tensor."""
        if self._writer is not None:
            self._writer.add_image(tag, image, step)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
