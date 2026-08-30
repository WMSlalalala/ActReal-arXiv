from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .utils import atomic_torch_save, load_json, now_tag, save_json


class CheckpointManager:
    def __init__(self, run_dir: Path, minimize_metric: bool = True):
        self.run_dir = Path(run_dir)
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.minimize_metric = bool(minimize_metric)
        self.manifest_path = self.ckpt_dir / "best_manifest.json"
        self.best_metric: Optional[float] = None
        self.best_path: Optional[Path] = None
        self.post_adv_manifest_path = self.ckpt_dir / "best_post_adv_manifest.json"
        self.best_post_adv_metric: Optional[float] = None
        self.best_post_adv_path: Optional[Path] = None
        if self.manifest_path.exists():
            try:
                m = load_json(str(self.manifest_path))
                self.best_metric = float(m["best_metric"])
                self.best_path = Path(m["best_path"])
            except Exception:
                self.best_metric = None
                self.best_path = None
        if self.post_adv_manifest_path.exists():
            try:
                m = load_json(str(self.post_adv_manifest_path))
                self.best_post_adv_metric = float(m["best_metric"])
                self.best_post_adv_path = Path(m["best_path"])
            except Exception:
                self.best_post_adv_metric = None
                self.best_post_adv_path = None

    def _state(
        self,
        epoch: int,
        global_step: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        ema: Any,
        cfg: Dict[str, Any],
        action: str,
        protocol: str,
        normalizer: Any,
        metrics: Dict[str, float],
        adv: Optional[torch.nn.Module] = None,
        adv_optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "ema_state": ema.state_dict() if ema is not None else None,
            "config": cfg,
            "action": action,
            "protocol": protocol,
            "normalizer": normalizer.to_dict() if normalizer is not None else None,
            "metrics": metrics,
        }
        if adv is not None:
            state["adv_state"] = adv.state_dict()
        if adv_optimizer is not None:
            state["adv_optimizer_state"] = adv_optimizer.state_dict()
        return state

    def save_last(self, **kwargs: Any) -> Path:
        path = self.ckpt_dir / "last.pt"
        atomic_torch_save(self._state(**kwargs), path)
        return path

    def save_epoch(self, epoch: int, **kwargs: Any) -> Path:
        path = self.ckpt_dir / ("epoch_%04d.pt" % int(epoch))
        if not path.exists():
            atomic_torch_save(self._state(epoch=epoch, **kwargs), path)
        return path

    def _is_better(self, metric: float) -> bool:
        if self.best_metric is None:
            return True
        if self.minimize_metric:
            return float(metric) < float(self.best_metric)
        return float(metric) > float(self.best_metric)

    def save_best_if_needed(self, metric: float, epoch: int, global_step: int, **kwargs: Any) -> Optional[Path]:
        metric = float(metric)
        if not self._is_better(metric):
            return None
        filename = "best_epoch%04d_step%d_metric%.6f_%s.pt" % (int(epoch), int(global_step), metric, now_tag())
        path = self.ckpt_dir / filename
        suffix = 1
        while path.exists():
            path = self.ckpt_dir / (filename.replace(".pt", "_%d.pt" % suffix))
            suffix += 1
        atomic_torch_save(self._state(epoch=epoch, global_step=global_step, **kwargs), path)
        self.best_metric = metric
        self.best_path = path
        save_json(
            self.manifest_path,
            {
                "best_metric": metric,
                "best_path": str(path),
                "epoch": int(epoch),
                "global_step": int(global_step),
                "all_best_files_are_preserved": True,
            },
        )
        return path

    def save_best_post_adv_if_needed(
        self,
        metric: float,
        epoch: int,
        global_step: int,
        **kwargs: Any,
    ) -> Optional[Path]:
        """Save the best checkpoint among epochs where ADV is active."""
        metric = float(metric)
        if self.best_post_adv_metric is not None:
            better = metric < self.best_post_adv_metric if self.minimize_metric else metric > self.best_post_adv_metric
            if not better:
                return None
        filename = "best_post_adv_epoch%04d_step%d_metric%.6f_%s.pt" % (
            int(epoch),
            int(global_step),
            metric,
            now_tag(),
        )
        path = self.ckpt_dir / filename
        suffix = 1
        while path.exists():
            path = self.ckpt_dir / (filename.replace(".pt", "_%d.pt" % suffix))
            suffix += 1
        atomic_torch_save(self._state(epoch=epoch, global_step=global_step, **kwargs), path)
        self.best_post_adv_metric = metric
        self.best_post_adv_path = path
        save_json(
            self.post_adv_manifest_path,
            {
                "best_metric": metric,
                "best_path": str(path),
                "epoch": int(epoch),
                "global_step": int(global_step),
                "selection_scope": "adv_active_epochs_only",
                "all_best_files_are_preserved": True,
            },
        )
        return path


def resolve_checkpoint(run_dir: Path, selector: str) -> Path:
    sel = str(selector)
    p = Path(sel)
    if p.exists():
        return p
    ckpt_dir = Path(run_dir) / "checkpoints"
    if sel == "last":
        out = ckpt_dir / "last.pt"
        if not out.exists():
            raise FileNotFoundError(str(out))
        return out
    if sel == "best":
        manifest = ckpt_dir / "best_manifest.json"
        if not manifest.exists():
            raise FileNotFoundError("best checkpoint manifest not found: %s" % manifest)
        m = load_json(str(manifest))
        out = Path(m["best_path"])
        if not out.exists():
            raise FileNotFoundError("best checkpoint listed in manifest does not exist: %s" % out)
        return out
    if sel == "best_post_adv":
        manifest = ckpt_dir / "best_post_adv_manifest.json"
        if not manifest.exists():
            raise FileNotFoundError("best post-ADV checkpoint manifest not found: %s" % manifest)
        m = load_json(str(manifest))
        out = Path(m["best_path"])
        if not out.exists():
            raise FileNotFoundError("best post-ADV checkpoint listed in manifest does not exist: %s" % out)
        return out
    raise FileNotFoundError("unknown checkpoint selector/path: %s" % selector)
