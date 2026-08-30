from __future__ import annotations

import os
import re
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceAudit:
    requested: str
    physical_index: int
    name: str
    uuid: str | None
    visible_devices: str | None


def require_physical_gpu(
    device: str,
) -> tuple[torch.device, DeviceAudit]:
    """Fail closed unless ``device`` names an unambiguously physical GPU."""

    match = re.fullmatch(r"cuda:(0|[1-9][0-9]*)", device)
    if match is None:
        raise RuntimeError(
            f"Formal GPU audit requires an explicit cuda:N device, got {device!r}"
        )
    physical_index = int(match.group(1))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device_count = torch.cuda.device_count()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    identity_order = ",".join(map(str, range(device_count)))
    if visible not in (None, "", identity_order):
        raise RuntimeError(
            "Formal GPU audit rejects CUDA_VISIBLE_DEVICES remapping or hiding; "
            "request the physical cuda:N device directly."
        )
    if physical_index >= device_count:
        raise RuntimeError(
            f"Requested physical GPU {physical_index}, but only {device_count} "
            "CUDA devices are visible"
        )
    resolved = torch.device(device)
    torch.cuda.set_device(resolved)
    props = torch.cuda.get_device_properties(resolved)
    uuid = getattr(props, "uuid", None)
    audit = DeviceAudit(
        requested=device,
        physical_index=physical_index,
        name=props.name,
        uuid=str(uuid) if uuid is not None else None,
        visible_devices=visible,
    )
    return resolved, audit


def require_physical_gpu1(device: str = "cuda:1") -> tuple[torch.device, DeviceAudit]:
    """Fail closed unless the process is explicitly using physical CUDA device 1.

    CUDA_VISIBLE_DEVICES remapping is rejected because it makes logical cuda:0
    ambiguous in a multi-GPU formal run.
    """

    if device != "cuda:1":
        raise RuntimeError(f"Only physical cuda:1 is authorized, got {device!r}")
    return require_physical_gpu(device)
