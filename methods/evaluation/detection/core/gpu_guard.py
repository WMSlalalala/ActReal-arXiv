"""Detection-facing imports for the shared physical-GPU guard."""

from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from methods.shared.gpu_guard import (  # noqa: E402,F401
    DeviceAudit,
    require_physical_gpu,
    require_physical_gpu1,
)
