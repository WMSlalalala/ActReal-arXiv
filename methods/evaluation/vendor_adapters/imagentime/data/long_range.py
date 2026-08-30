"""HMOG inertial windows, delivered in the shape ImagenTime's loader expects.

The repository ships `data/` as an empty package -- the authors distribute the
actual corpora as a Google Drive archive -- so `gen_dataloader` cannot import
anything here until this file exists.  That makes this the intended extension
point rather than a modification: `utils/utils_data.py` is untouched, and the
branch at line 142 that handles `fred_md` and its five siblings calls
`parse_datasets(name, batch_size, device, args)`, stacks whatever list comes
back, and overwrites `args.seq_len` from its shape.

So an HMOG action is routed through their pipeline by naming the dataset after
one of those six and returning our windows here.  Everything downstream -- the
80/20 split, the TensorDataset, both DataLoaders -- is theirs.

NORMALISATION.  Every corpus they ship reaches the model min-max scaled into
[0, 1], which is what EDM's hard-coded `sigma_data = 0.5` assumes.  Raw inertial
data is nowhere near that range (tap spans -2.26 to +14.33), so the same scaling
is applied here, per channel, over the flattened stack.  Nothing in their
pipeline ever inverts it, so the statistics are written beside the samples for
the sampler to undo.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

# Set by the driver before the loader runs.  An environment variable rather than
# an argument because the call site is inside the authors' code and its
# signature is fixed.
WINDOW_ENV = "IMAGENTIME_HMOG_WINDOWS"
STATS_ENV = "IMAGENTIME_HMOG_STATS"


def parse_datasets(dataset_name, batch_size, device, args):
    """Return the HMOG stack as the list of per-window tensors the branch stacks."""

    source = os.environ.get(WINDOW_ENV)
    if not source:
        raise RuntimeError(
            f"{WINDOW_ENV} is not set: this loader only serves HMOG windows, and "
            "the driver must name the .npy stack to read"
        )

    windows = np.load(source).astype(np.float32)
    if windows.ndim != 3:
        raise ValueError(f"expected (N, T, K), got {windows.shape}")

    flattened = windows.reshape(-1, windows.shape[2])
    low = flattened.min(axis=0)
    high = flattened.max(axis=0)
    span = np.where(high - low > 1e-12, high - low, 1.0)
    scaled = (windows - low) / span

    stats_path = os.environ.get(STATS_ENV)
    if stats_path:
        Path(stats_path).write_text(
            json.dumps({"low": low.tolist(), "span": span.tolist(),
                        "source": source, "windows": int(len(windows))})
        )

    tensor = torch.from_numpy(scaled.astype(np.float32))
    # The branch does torch.stack(list), so a list of (T, K) tensors is what it
    # wants; handing back the stacked array directly would add an axis.
    return [tensor[i] for i in range(len(tensor))]
