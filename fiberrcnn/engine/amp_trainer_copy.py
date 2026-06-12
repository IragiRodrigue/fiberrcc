"""
AMP + Gradient Accumulation Trainer
=====================================
A custom Detectron2 ``SimpleTrainer`` subclass that provides:

* Native ``torch.cuda.amp.GradScaler`` AMP (Automatic Mixed Precision)
* True gradient accumulation across N micro-batches before each optimiser step
* Per-step loss logging with TensorBoard
* Gradient clipping

This replaces Detectron2's standard ``SimpleTrainer`` when AMP or
gradient accumulation is needed beyond the default hooks.

Configuration keys (added to CfgNode):
  SOLVER.AMP.ENABLED          : bool   (default True)
  SOLVER.GRAD_ACCUMULATE_STEPS: int    (default 1)
  SOLVER.CLIP_GRADIENTS.ENABLED     : bool
  SOLVER.CLIP_GRADIENTS.CLIP_VALUE  : float
"""

from __future__ import annotations

import logging
import time
from typing import Any

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel

logger = logging.getLogger(__name__)


class AMPGradAccumTrainer:
    """AMP + gradient-accumulation training loop.

    Parameters
    ----------
    model : nn.Module
        The FiberRCNN model.
    data_loader : iterable
        Detectron2 DataLoader.
    optimizer : torch.optim.Optimizer
    amp_enabled : bool
        Use automatic mixed precision.
    accumulate_steps : int
        Number of micro-batches before ``optimizer.step()``.
    grad_clip_value : float | None
        Max gradient norm (None = no clipping).
    tb_writer : optional TensorBoard SummaryWriter
    """

    def __init__(
        self,
        model: nn.Module,
        data_loader,
        optimizer: torch.optim.Optimizer,
        amp_enabled: bool = True,
        accumulate_steps: int = 1,
        grad_clip_value: float | None = 1.0,
        tb_writer: Any | None = None,
    ) -> None:
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.amp_enabled = amp_enabled and torch.cuda.is_available()
        self.accumulate_steps = max(1, accumulate_steps)
        self.grad_clip_value = grad_clip_value
        self.tb_writer = tb_writer

        self.scaler = GradScaler(enabled=self.amp_enabled)
        self._data_iter = iter(data_loader)

        self.iter: int = 0
        self.start_iter: int = 0
        self.max_iter: int = 0

        logger.info(
            f"AMPGradAccumTrainer: AMP={self.amp_enabled}, "
            f"accumulate={self.accumulate_steps}, "
            f"grad_clip={self.grad_clip_value}"
        )

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def run_step(self) -> dict[str, float]:
        """Execute one effective optimiser step (= ``accumulate_steps`` forward passes).

        Returns
        -------
        loss_dict : averaged losses for this optimiser step
        """
        assert self.model.training, "Model must be in training mode."

        self.optimizer.zero_grad()

        accumulated_losses: dict[str, float] = {}
        total_loss_sum = 0.0

        for micro in range(self.accumulate_steps):
            try:
                data = next(self._data_iter)
            except StopIteration:
                self._data_iter = iter(self.data_loader)
                data = next(self._data_iter)

            with autocast(enabled=self.amp_enabled):
                loss_dict = self.model(data)
                losses = sum(loss_dict.values())
                # Scale by accumulate_steps to get the average gradient
                losses_scaled = losses / self.accumulate_steps

            self.scaler.scale(losses_scaled).backward()

            total_loss_sum += losses.item()
            for k, v in loss_dict.items():
                accumulated_losses[k] = accumulated_losses.get(k, 0.0) + v.item()

        # Average
        for k in accumulated_losses:
            accumulated_losses[k] /= self.accumulate_steps
        avg_total = total_loss_sum / self.accumulate_steps

        # Unscale + clip
        self.scaler.unscale_(self.optimizer)
        if self.grad_clip_value is not None:
            params = [
                p for group in self.optimizer.param_groups for p in group["params"]
            ]
            nn.utils.clip_grad_norm_(params, self.grad_clip_value)

        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Logging
        if self.tb_writer is not None:
            self.tb_writer.add_scalar("train/total_loss", avg_total, self.iter)
            for k, v in accumulated_losses.items():
                self.tb_writer.add_scalar(f"train/{k}", v, self.iter)

        return accumulated_losses

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        start_iter: int,
        max_iter: int,
        scheduler: Any | None = None,
        eval_fn: Any | None = None,
        eval_period: int = 5000,
        checkpoint_fn: Any | None = None,
        checkpoint_period: int = 5000,
    ) -> None:
        """Run training from ``start_iter`` to ``max_iter``.

        Parameters
        ----------
        start_iter, max_iter : iteration bounds
        scheduler : LR scheduler with a ``.step()`` method
        eval_fn : callable() called every ``eval_period`` iters
        checkpoint_fn : callable(iter) called every ``checkpoint_period`` iters
        """
        self.start_iter = start_iter
        self.max_iter = max_iter
        self.iter = start_iter

        self.model.train()
        logger.info(f"Starting training: iter {start_iter} → {max_iter}")

        start_time = time.perf_counter()

        for self.iter in range(start_iter, max_iter):
            loss_dict = self.run_step()

            if scheduler is not None:
                scheduler.step()

            # Periodic evaluation
            if eval_fn is not None and (self.iter + 1) % eval_period == 0:
                self.model.eval()
                with torch.no_grad():
                    eval_fn(self.iter + 1)
                self.model.train()

            # Periodic checkpoint
            if checkpoint_fn is not None and (self.iter + 1) % checkpoint_period == 0:
                checkpoint_fn(self.iter + 1)

            # Console log every 20 iters
            if (self.iter + 1) % 20 == 0:
                elapsed = time.perf_counter() - start_time
                its_per_s = (self.iter - start_iter + 1) / max(elapsed, 1e-6)
                loss_str = "  ".join(f"{k}={v:.4f}" for k, v in loss_dict.items())
                logger.info(
                    f"[{self.iter + 1}/{max_iter}]  {loss_str}  "
                    f"({its_per_s:.1f} it/s)"
                )

        total_time = time.perf_counter() - start_time
        logger.info(
            f"Training complete in {total_time / 60:.1f} min "
            f"({(max_iter - start_iter) / total_time:.1f} it/s avg)"
        )

    # ------------------------------------------------------------------
    # State dict for checkpointing
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iter,
            "scaler": self.scaler.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.iter = state.get("iteration", 0)
        if "scaler" in state:
            self.scaler.load_state_dict(state["scaler"])
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])


# ---------------------------------------------------------------------------
# Helper: build optimiser with layer-wise LR decay (useful for transformers)
# ---------------------------------------------------------------------------

def build_optimizer_with_layer_lr_decay(
    model: nn.Module,
    base_lr: float = 0.0025,
    weight_decay: float = 1e-4,
    layer_decay: float = 0.75,
    backbone_prefix: str = "backbone",
) -> torch.optim.AdamW:
    """Build AdamW with layer-wise LR decay for transformer backbones.

    Layers closer to the input receive lower learning rates, following
    the strategy from BEiT / MAE.

    Parameters
    ----------
    model : FiberRCNN model
    base_lr : learning rate for the output head
    weight_decay : weight decay coefficient
    layer_decay : multiplicative decay per layer (0.75 = 75% of next layer's LR)
    backbone_prefix : parameter name prefix for the backbone

    Returns
    -------
    torch.optim.AdamW
    """
    param_groups: list[dict[str, Any]] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # No weight decay on biases and norms
        wd = 0.0 if ("bias" in name or "norm" in name or "bn" in name) else weight_decay

        # Layer depth: rough heuristic based on name depth
        depth = name.count(".")
        lr = base_lr * (layer_decay ** depth) if backbone_prefix in name else base_lr

        param_groups.append({"params": [param], "lr": lr, "weight_decay": wd})

    return torch.optim.AdamW(param_groups, lr=base_lr)
