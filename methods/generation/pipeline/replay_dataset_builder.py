from __future__ import annotations

"""Build a leakage-safe trajectory replacement dataset.

The builder is intentionally non-destructive: it reads an existing direct100k
ragged dataset, retains its IMU samples and event identities, and writes a new
dataset directory.  Genuine and replayed touch rows pass through the same
Android zero-order-hold observer.

Two explicit modes are exposed.  The small smoke mode uses a frozen
train-fitted conditional generator for fake tap, scroll, and swipe touch rows.
The full mode preserves its existing replay behavior until that generator has
passed the smoke non-regression gates.  This distinction is recorded in event
provenance and never misrepresented as per-event human donor replay.
"""

import concurrent.futures
import functools
import hashlib
import json
import math
import multiprocessing
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .action_replay import (
    DEFAULT_MAX_TIME_WARP,
    ActionReplayBank,
    ActionReplayError,
    DonorSplitPools,
    IsometricReplayAllocation,
    ReplayAllocator,
    ReplayGeometry,
    _fitted_isometries,
    classify_replay_request,
    observe_replay_primitive,
)
from .android_touch_observation import (
    TouchObservationError,
    ACTION_CANCEL,
    ACTION_DOWN,
    ACTION_MASK,
    ACTION_MOVE,
    ACTION_UP,
    TouchObservation,
    detector_grid_clock,
    detector_grid_span_ms,
    observe_android_rows,
    screen_dimensions_for_orientation,
)
from .audit import sha256_file
from .conditional_touch_generator import (
    ConditionalTouchGenerator,
    ConditionalTouchGeneratorError,
)
from .conditional_touch_request_generator import (
    ConditionalTouchRequestGenerator,
    ConditionalTouchRequestGeneratorError,
    IMPORT_SOURCE_FINGERPRINT_SHA256 as CONDITIONAL_TOUCH_REQUEST_SOURCE_FINGERPRINT_SHA256,
)
from .conditional_touch_request_plan import ConditionalTouchRequestPlan
from .event_pad import (
    IMU_CHANNELS,
    EventPadError,
    _load_manifest,
    load_event_partition,
)
from .exact_touch_template_generator import (
    DIRECTION8 as EXACT_TOUCH_DIRECTION8,
    ExactTouchTemplateError,
    SCHEMA_VERSION as EXACT_TOUCH_TEMPLATE_SCHEMA_VERSION,
    STATIONARY as EXACT_TOUCH_STATIONARY,
    generate_exact_touch_template,
)
from .fiveshot_gesture_timing import (
    FiveShotGestureTimingError,
    GestureDurationLaw,
    carrier_window_imu,
    contact_travel_px,
    law_from_material,
)
from .fiveshot_material import (
    FiveShotMaterialPool,
    FiveShotMaterialError,
    MaterialEvent,
    load_fiveshot_material,
)
from .genuine_touch_recovery import (
    GenuineTouchBinding,
    GenuineTouchRecoveryError,
    load_genuine_touch_bindings,
    load_raw_trajectory_archive,
    observe_genuine_binding,
)
from .keystroke_replay import (
    KeystrokeReplayError,
    TimingBounds,
)
from .keystroke_imu_pulse import (
    KeystrokeImuPulseError,
    KeystrokeImuPulseModel,
    fit_keystroke_imu_pulse_model,
    load_user_pulse_sources,
    generate_keystroke_imu,
)
from .fiveshot_keystroke_touch import (
    FiveShotKeystrokeTouchError,
    KeystrokeRhythm,
    compose_keystroke_touch,
    rhythm_from_material,
)
from .pinch_endpoint_control import (
    PinchEndpointControlError,
    PinchEndpointGeometry,
    extract_live_two_pointer_endpoints,
)


ACTIONS = ("tap", "scroll", "swipe", "pinch", "keystroke")
SPLITS = ("train", "development", "test")
DETECTOR_MODALITIES = (
    "trajectory_xytime",
    "imu_only",
    "imu_trajectory_xytime",
)
INPUT_SHARD_SCHEMA = "joint_event_pad_ragged_shard_v1"
INPUT_MANIFEST_SCHEMA = "joint_event_pad_manifest_v2"
DATASET_RELEASE_SCHEMA = "hmog_direct100k_detector_dataset_v1"
REBUILDER_SCHEMA = "hmog_replay_trajectory_rebuilder_v1"
PROVENANCE_SCHEMA = "hmog_replay_trajectory_event_provenance_v1"
# Smoke is a release mode, not an Event PAD manifest scope.  Reuse the
# registered small-data scope so the exact detector loader/cell contract stays
# valid without widening the formal protocol's scope registry.
SMOKE_MANIFEST_SCOPE = "balanced_small"

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_ROOT = REPO_ROOT / "licensed_input" / "hmog" / "processed_trajectories"
DEFAULT_SPLIT_PATH = REPO_ROOT / "data" / "splits" / "users_seed42.json"

# Frozen 5--95% train support.  Targets outside this support are rejected and
# replaced by another smoke candidate rather than stretched into a tail event.
ROBUST_KEYSTROKE_BOUNDS = TimingBounds()
SINGLE_POINTER_TOUCH_ACTIONS = frozenset(("tap", "scroll", "swipe"))
# All three single-pointer actions now transform a five-shot donor onto the
# requested chord.  Scroll used to be generated from a fitted statistical model
# instead, which made it the only action carrying no real human substrate -- and
# measurably the easiest to detect (trajectory-only AUC 0.73 against 0.52-0.58
# for the donor-based actions).
EXACT_TOUCH_TEMPLATE_ACTIONS = frozenset(("tap", "scroll", "swipe"))
CONDITIONAL_TOUCH_ACTIONS: frozenset[str] = frozenset()
CONDITIONAL_TOUCH_REBUILD_METHOD = "conditional_touch_generator_model"
EXACT_TOUCH_TEMPLATE_REBUILD_METHOD = "exact_touch_template_generator_v1"
# Pinch is requested as an area plus a magnitude, never as four endpoints, so it
# gets its own single-similarity route instead of the endpoint-pair transform.
FIVESHOT_SHOTS_PER_GROUP = 5
# The release gives every user 200 fake events per action, so one of five frozen
# shots carries forty of them.  A smaller build rebuilds a subset of the same
# plan and keeps the same cap.
FIVESHOT_FAKE_EVENTS_PER_GROUP = 200
FIVESHOT_KEYSTROKE_TOUCH_METHOD = "fiveshot_keystroke_rhythm_touch"
FIVESHOT_KEYSTROKE_GENERATION_MODE = "fiveshot_keystroke_rhythm_v1"
PINCH_AREA_SIMILARITY_REBUILD_METHOD = "fiveshot_pinch_area_similarity"
PINCH_AREA_SIMILARITY_GENERATION_MODE = "pinch_area_similarity_v1"
PINCH_AREA_SIMILARITY_ACTIONS = frozenset(("pinch",))
PINCH_AREA_REQUEST_SOURCES = frozenset(
    ("frozen_fiveshot_material_requested_area",)
)
# An area has no preferred diagonal end, so either assignment of the two
# requested points to the two fingers satisfies the same request.
PINCH_POINTER_ORDERS = ("as_requested", "swapped")
# Every transform the exact-touch generator can declare.  Only the identity
# mode leaves the donor bytes untouched; the rest move a five-shot donor onto
# the requested chord.  The generator itself already refuses to return any of
# them unless the endpoints, duration and direction land exactly, so the
# provenance check below constrains the mode name rather than re-deriving the
# geometry.
EXACT_TOUCH_TRANSFORM_MODES = frozenset(
    (
        "identity_template",
        "tap_translation",
        "tap_residual_bridge",
        "tap_chord_frame",
        "scroll_chord_frame",
        "swipe_chord_frame",
    )
)
EXACT_TOUCH_REQUEST_SOURCES = frozenset(
    (
        "frozen_smoke_reference_exact_endpoints",
        "frozen_fiveshot_material_exact_endpoints",
        "frozen_fiveshot_material_donor_drift_endpoints",
    )
)
FIVESHOT_MATERIAL_REQUEST_SOURCES = frozenset(
    (
        "frozen_fiveshot_material_exact_endpoints",
        "frozen_fiveshot_material_donor_drift_endpoints",
    )
)
# Android keeps dispatching a one-finger gesture as a tap until the pointer
# leaves a touch-slop radius around its DOWN point; every generated carrier
# target records that radius as 24 px, and the frozen material audit already
# uses the same number to exclude donors whose endpoints moved further.  A tap
# request may therefore carry its donor's own drift up to this budget without
# changing which view the gesture reaches.
FIVESHOT_TAP_DRIFT_LIMIT_PX = 24.0
FIVESHOT_TAP_SCREEN_MARGIN_PX = 1.0e-3
# Actions whose fake touch is injected as (time, point) updates on the victim's
# own report cadence and observed once, instead of being index-resampled onto the
# carrier's row count.  Set by the environment so two candidates can be built and
# scored side by side; empty means every action keeps the resampler.
SYNTHESISED_CLOCK_ACTIONS = frozenset(
    value
    for value in os.environ.get("HMOG_SYNTHESISED_CLOCK_ACTIONS", "").split(",")
    if value
)
# Actions whose fake gesture takes as long as the victim's own five recordings
# say a gesture of that travel takes, instead of as long as the carrier happened
# to choose.  See `fiveshot_gesture_timing` for what the carrier's choice costs.
# Set by the environment so candidates can be built and scored side by side;
# empty means every action keeps the carrier's duration.
FIVESHOT_TIMING_ACTIONS = frozenset(
    value
    for value in os.environ.get("HMOG_FIVESHOT_TIMING_ACTIONS", "").split(",")
    if value
)
# The detector grid the observer reports on, so a duration in milliseconds and a
# row count are the same statement.
FIVESHOT_TIMING_PERIOD_MS = 10.0
# Whether the timed gesture also carries the spread the victim's own material
# shows around its own curve.  Reading every duration off the curve makes the
FIVESHOT_TIMING_SPREAD = os.environ.get(
    "HMOG_FIVESHOT_TIMING_SPREAD", "none"
).strip()
FIVESHOT_TIMING_SPREAD_POLICIES = frozenset({"none", "loo_residual"})
# The shortest gesture the detector grid can carry: a DOWN, a move and an UP is
# three rows, which is 20 ms.  A departure applied at the short end of the curve
# can ask for less than that -- 0.24% of the timed scrolls do -- and this is the
# floor those are reported at, recorded per event rather than silently applied.
FIVESHOT_TIMING_FLOOR_MS = 20.0
if FIVESHOT_TIMING_SPREAD not in FIVESHOT_TIMING_SPREAD_POLICIES:
    raise ValueError(
        "HMOG_FIVESHOT_TIMING_SPREAD must be one of "
        f"{sorted(FIVESHOT_TIMING_SPREAD_POLICIES)}"
    )


class ReplayDatasetBuildError(RuntimeError):
    pass


def _is_hmog_ascii_letter_keycode(value: int) -> bool:
    """Return the frozen HMOG letter flag for a raw keycode token."""

    code = int(value)
    return 65 <= code <= 90 or 97 <= code <= 122


@dataclass(frozen=True)
class InputShard:
    split: str
    user_id: str
    path: Path
    sha256: str
    manifest_row: Mapping[str, Any]


@dataclass(frozen=True)
class AndroidTarget:
    action: str
    orientation_id: int
    direction: str | None
    pinch_scale_direction: str | None
    keycodes: tuple[int, ...]
    trajectory_source: Path
    trajectory_source_sha256: str
    trajectory_archive_index: int
    raw_duration_ms: float
    t_ms: np.ndarray
    x_px: np.ndarray
    y_px: np.ndarray
    pressure: np.ndarray
    pointer_id: np.ndarray
    android_action: np.ndarray
    key_index: np.ndarray
    frame_index: np.ndarray
    frame_end: np.ndarray
    bound_event_plan_sha256: str


@dataclass(frozen=True)
class DurationRatioSample:
    raw_to_window_ratio: float
    target_update_rate_hz: float | None
    reference_source_cluster_id: str | None
    reference_key_count: int | None
    conditioning: str
    conditioning_count: int
    reference_raw_duration_ms: float | None = None
    reference_window_duration_ms: float | None = None
    reference_window_sample_count: int | None = None
    reference_observable_update_count: int | None = None


@dataclass(frozen=True)
class RawTimingReference:
    raw_to_window_ratio: float
    observable_update_rate_hz: float | None
    source_cluster_id: str | None
    key_count: int | None = None
    raw_duration_ms: float | None = None
    window_duration_ms: float | None = None
    window_sample_count: int | None = None
    observable_update_count: int | None = None


@dataclass(frozen=True)
class KeystrokeReferenceTemplate:
    source_cluster_id: str
    source_event_id: str
    source_user_id: int
    source_session_id: int
    orientation_id: int
    keycodes: tuple[int, ...]
    down_anchors_px: tuple[tuple[float, float], ...]
    raw_trajectory_source: Path
    raw_trajectory_source_sha256: str
    raw_trajectory_event_index: int
    raw_event_sha256: str


@dataclass(frozen=True)
class KeystrokeTargetPlan:
    """One coupled human key-count/geometry/timing carrier target."""

    timing: DurationRatioSample
    template: KeystrokeReferenceTemplate
    raw_touch_duration_ms: int
    detector_duration_ms: float
    target_samples: int
    letter_count: int


@dataclass(frozen=True)
class _GestureTiming:
    """How long one fake gesture takes, and where its inertia comes from."""

    travel_px: float
    duration_ms: float
    samples: int
    requested_samples: int
    imu_source: str
    law_source_event_ids: tuple[str, ...]
    spread_policy: str = "none"
    log_offset: float = 0.0
    residual_draw: int = -1
    residual_index: int = -1
    residual_spread: float = 0.0
    floored_to_reportable: bool = False

    @property
    def capped_to_window(self) -> bool:
        """Report whether the carrier's window was shorter than the law asked."""

        return self.samples != self.requested_samples


@dataclass(frozen=True)
class TapDonor:
    binding: GenuineTouchBinding
    raw_duration_ms: float


@dataclass
class LoadedShard:
    source: InputShard
    arrays: dict[str, np.ndarray]

    @property
    def event_count(self) -> int:
        return len(self.arrays["label"])

    def event_signal(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        offsets = self.arrays["offsets"]
        left, right = (int(value) for value in offsets[index : index + 2])
        return (
            np.asarray(self.arrays["imu_flat"][left:right], dtype=np.float32),
            np.asarray(
                self.arrays["trajectory_flat"][left:right], dtype=np.float32
            ),
        )


@dataclass(frozen=True)
class RebuiltEventSignal:
    """One rebuilt paired event; ragged length may change as one unit."""

    imu: np.ndarray
    trajectory: np.ndarray


class TapReplayAllocator:
    """Train-only, event-family partitioned, without-replacement tap replay."""

    def __init__(
        self,
        donors: Sequence[TapDonor],
        *,
        output_split: str,
        split_seed: int,
    ) -> None:
        if output_split not in SPLITS:
            raise ReplayDatasetBuildError(f"unknown tap output split {output_split}")
        self.output_split = output_split
        selected = [
            donor
            for donor in donors
            if _donor_output_split(
                f"tap|{donor.binding.source_event_id}", seed=split_seed
            )
            == output_split
        ]
        self._remaining = sorted(
            selected,
            key=lambda donor: (
                donor.binding.orientation_id,
                donor.raw_duration_ms,
                donor.binding.source_event_id,
            ),
        )
        self.used_source_event_ids: set[str] = set()

    def allocate(
        self,
        *,
        orientation_id: int,
        target_duration_ms: float,
        max_time_warp: float = DEFAULT_MAX_TIME_WARP,
    ) -> TapDonor:
        candidates: list[tuple[float, str, int, TapDonor]] = []
        for index, donor in enumerate(self._remaining):
            binding = donor.binding
            if binding.orientation_id != int(orientation_id):
                continue
            ratio = float(target_duration_ms) / donor.raw_duration_ms
            if 1.0 / max_time_warp <= ratio <= max_time_warp:
                candidates.append(
                    (
                        abs(donor.raw_duration_ms - float(target_duration_ms)),
                        binding.source_event_id,
                        index,
                        donor,
                    )
                )
        if not candidates:
            raise ReplayDatasetBuildError(
                "tap donor pool has no duration/orientation-compatible raw contact"
            )
        _, _, index, donor = min(candidates)
        self._remaining.pop(index)
        binding = donor.binding
        if binding.source_event_id in self.used_source_event_ids:
            raise AssertionError("tap allocator reused a donor")
        self.used_source_event_ids.add(binding.source_event_id)
        return donor


def _donor_output_split(identity: str, *, seed: int) -> str:
    digest = hashlib.sha256(f"donor|{seed}|{identity}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") % 100
    if value < 70:
        return "train"
    if value < 80:
        return "development"
    return "test"


def _event_seed(event_id: str, *, seed: int) -> int:
    digest = hashlib.sha256(
        f"replay-event|{int(seed)}|{event_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _conditional_touch_request_seed(event_id: str, *, seed: int) -> int:
    digest = hashlib.sha256(
        f"conditional-touch-request|{int(seed)}|{event_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _pair_id(event_id: str, imu: np.ndarray, trajectory: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(b"hmog-replay-pair-v1\0")
    digest.update(str(event_id).encode("utf-8"))
    digest.update(bytes.fromhex(_array_sha256(imu)))
    digest.update(bytes.fromhex(_array_sha256(trajectory)))
    return digest.hexdigest()


def _detector_window_duration_ms(
    trajectory: np.ndarray,
    *,
    fallback_duration_ms: float | None = None,
) -> float:
    values = np.asarray(trajectory, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 9 or len(values) < 2:
        raise ReplayDatasetBuildError("detector trajectory shape is invalid")
    elapsed = values[:, 7] * 1000.0
    duration = float(elapsed[-1])
    if (
        not np.isfinite(elapsed).all()
        or abs(float(elapsed[0])) > 1.0e-4
        or np.any(np.diff(elapsed) < -1.0e-4)
    ):
        raise ReplayDatasetBuildError("detector trajectory time endpoint is invalid")
    if duration <= 0.0:
        fallback = (
            None if fallback_duration_ms is None else float(fallback_duration_ms)
        )
        if (
            not np.all(np.abs(elapsed) <= 1.0e-4)
            or fallback is None
            or not np.isfinite(fallback)
            or fallback <= 0.0
        ):
            raise ReplayDatasetBuildError(
                "detector trajectory time endpoint is invalid"
            )
        return fallback
    return duration


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReplayDatasetBuildError(f"{path}: expected a JSON object")
    return value


def _read_jsonl_objects(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ReplayDatasetBuildError(
                    f"{path}:{line_number}: expected JSON object"
                )
            yield value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _resolve_shard_path(
    manifest_path: Path, shard: Mapping[str, Any]
) -> Path:
    declared = Path(str(shard.get("source", ""))).resolve()
    fallback = manifest_path.parent / "shards" / declared.name
    if declared.is_file():
        return declared
    if fallback.is_file():
        return fallback.resolve()
    raise ReplayDatasetBuildError(f"input shard is missing: {declared}")


def load_input_dataset(
    manifest_path: str | Path,
    release_path: str | Path | None = None,
) -> tuple[dict[str, list[InputShard]], dict[str, Any]]:
    manifest = Path(manifest_path).resolve()
    release_source = (
        manifest.parent / "release.json"
        if release_path is None
        else Path(release_path).resolve()
    )
    if not manifest.is_file() or not release_source.is_file():
        raise ReplayDatasetBuildError("input manifest/release is missing")
    release = _read_json(release_source)
    if release.get("schema_version") != DATASET_RELEASE_SCHEMA:
        raise ReplayDatasetBuildError("input is not a direct100k detector release")
    declared_manifest_digest = str(release.get("event_manifest_sha256", ""))
    if declared_manifest_digest and declared_manifest_digest != sha256_file(manifest):
        raise ReplayDatasetBuildError("input release does not bind the manifest bytes")
    rows: dict[str, list[InputShard]] = {}
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                not isinstance(row, dict)
                or row.get("schema_version") != INPUT_MANIFEST_SCHEMA
                or str(row.get("split", "")) not in SPLITS
                or not isinstance(row.get("shards"), list)
            ):
                raise ReplayDatasetBuildError(
                    f"{manifest}:{line_number}: invalid sharded partition"
                )
            split = str(row["split"])
            if split in rows:
                raise ReplayDatasetBuildError(f"duplicate manifest split {split}")
            parts: list[InputShard] = []
            for shard in row["shards"]:
                path = _resolve_shard_path(manifest, shard)
                digest = sha256_file(path)
                if digest != str(shard.get("source_sha256", "")):
                    raise ReplayDatasetBuildError(f"input shard hash changed: {path}")
                parts.append(
                    InputShard(
                        split=split,
                        user_id=str(shard.get("user_id", "")),
                        path=path,
                        sha256=digest,
                        manifest_row=dict(shard),
                    )
                )
            rows[split] = parts
    if set(rows) != set(SPLITS):
        raise ReplayDatasetBuildError("input manifest needs three fixed splits")
    return rows, release


def _load_shard(source: InputShard, *, signals: bool = True) -> LoadedShard:
    required = {
        "schema_version",
        "coordinate_schema",
        "time_schema",
        "scope",
        "split",
        "imu_flat",
        "trajectory_flat",
        "offsets",
        "label",
        "user_id",
        "session_id",
        "event_id",
        "action",
        "source_cluster_id",
        "sample_idx",
        "cross_modal_pair_id",
    }
    with np.load(source.path, allow_pickle=False) as archive:
        missing = required - set(archive.files)
        if missing:
            raise ReplayDatasetBuildError(
                f"{source.path}: missing shard arrays {sorted(missing)}"
            )
        arrays = {
            name: np.asarray(archive[name]).copy()
            for name in required
            if signals or name not in {"imu_flat", "trajectory_flat"}
        }
    if (
        str(np.asarray(arrays["schema_version"]).item()) != INPUT_SHARD_SCHEMA
        or str(np.asarray(arrays["split"]).item()) != source.split
    ):
        raise ReplayDatasetBuildError(f"{source.path}: shard contract mismatch")
    return LoadedShard(source=source, arrays=arrays)


def _selected_genuine_windows(
    shards_by_split: Mapping[str, Sequence[InputShard]],
    genuine_bindings: Mapping[str, GenuineTouchBinding],
) -> tuple[dict[str, float], dict[str, int], int]:
    selected: dict[str, float] = {}
    sample_counts: dict[str, int] = {}
    repaired = 0
    raw_cache: dict[Path, dict[str, np.ndarray]] = {}
    for split in SPLITS:
        for source in shards_by_split[split]:
            shard = _load_shard(source, signals=True)
            labels = np.asarray(shard.arrays["label"], dtype=np.int64)
            clusters = np.asarray(shard.arrays["source_cluster_id"]).astype(str)
            for index in np.flatnonzero(labels == 0):
                cluster = str(clusters[index])
                _, trajectory = shard.event_signal(int(index))
                binding = genuine_bindings.get(cluster)
                if binding is None:
                    raise ReplayDatasetBuildError(
                        f"selected genuine has no raw binding: {cluster}"
                    )
                old_endpoint = float(trajectory[-1, 7] * 1000.0)
                recovery_duration: float | None = None
                if old_endpoint <= 0.0:
                    archive = _raw_archive_cached(binding, raw_cache)
                    recovery_duration = _binding_raw_duration_and_request(
                        binding, archive
                    )[0]
                duration_ms = _detector_window_duration_ms(
                    trajectory,
                    fallback_duration_ms=recovery_duration,
                )
                if old_endpoint <= 0.0:
                    repaired += 1
                if duration_ms <= 0.0 or cluster in selected:
                    raise ReplayDatasetBuildError(
                        "selected genuine detector-window identity is invalid"
                    )
                selected[cluster] = duration_ms
                sample_counts[cluster] = int(len(trajectory))
    return selected, sample_counts, repaired


def _load_trajectory_archive(
    path: Path,
    cache: dict[Path, dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    source = path.resolve()
    if source in cache:
        return cache[source]
    required = {
        "action",
        "duration_ms",
        "orientation_id",
        "user_id",
        "split_id",
        "sample_index",
        "event_plan_sha256",
        "clipped_point_count",
        "clipped_point_rate",
        "geometry_valid",
        "geometry_outlier",
        "geometry_exclusion_code",
        "pre_projection_oob_point_count",
        "pre_projection_oob_point_rate",
        "typed_target_dispatch_feasibility_gate_pass",
        "generated_dispatch_quality_gate_pass",
        "target_clipping_applied",
        "slot_dropped",
        "physical_clip_count",
        "physical_clip_rate",
        "physical_clipped_coordinate_value_count",
        "android_offsets",
        "flat_android_t_ms",
        "flat_android_x",
        "flat_android_y",
        "flat_android_pressure",
        "flat_android_pointer_id",
        "flat_android_action",
        "flat_android_key_index",
        "flat_android_keycode",
        "flat_android_frame_index",
        "flat_android_frame_end",
    }
    with np.load(source, allow_pickle=False) as archive:
        missing = required - set(archive.files)
        if missing:
            raise ReplayDatasetBuildError(
                f"{source}: incomplete fake trajectory source {sorted(missing)}"
            )
        values = {name: np.asarray(archive[name]).copy() for name in required}
    # Cache the digest alongside the decoded archive.  Large builds therefore
    # hash each immutable source once while every event binding remains
    # cryptographically checked.
    values["__source_sha256"] = np.asarray(sha256_file(source))
    cache[source] = values
    return values


def load_android_target(
    *,
    event_id: str,
    action: str,
    target_duration_ms: float,
    joint_events_root: Path,
    trajectory_cache: dict[Path, dict[str, np.ndarray]],
    expected_user_id: str | None = None,
    expected_split: str | None = None,
    expected_sample_idx: int | None = None,
    expected_cross_modal_pair_id: str | None = None,
) -> AndroidTarget:
    joint_path = joint_events_root / f"{event_id}.npz"
    if not joint_path.is_file():
        raise ReplayDatasetBuildError(f"joint fake event is missing: {joint_path}")
    with np.load(joint_path, allow_pickle=False) as archive:
        required = {
            "source_action_label",
            "trajectory_source",
            "trajectory_source_sha256",
            "trajectory_archive_index",
            "orientation_id",
            "paired_sample_index",
            "shared_event_plan_sha256",
            "physical_out_of_bounds_point_count",
            "physical_out_of_bounds_point_rate",
            "physical_clip_count",
            "physical_clip_rate",
            "physical_clipped_coordinate_value_count",
            "typed_target_dispatch_feasibility_gate_pass",
            "generated_dispatch_quality_gate_pass",
            "target_clipping_applied",
            "slot_dropped",
        }
        if required - set(archive.files):
            raise ReplayDatasetBuildError(f"{joint_path}: incomplete joint binding")
        observed_action = str(np.asarray(archive["source_action_label"]).item())
        trajectory_source = Path(
            str(np.asarray(archive["trajectory_source"]).item())
        ).resolve()
        trajectory_source_sha256 = str(
            np.asarray(archive["trajectory_source_sha256"]).item()
        )
        archive_index = int(np.asarray(archive["trajectory_archive_index"]).item())
        orientation = int(np.asarray(archive["orientation_id"]).item())
        paired_sample_index = int(
            np.asarray(archive["paired_sample_index"]).item()
        )
        shared_value = np.asarray(archive["shared_event_plan_sha256"])
        if shared_value.ndim == 0 and isinstance(shared_value.item(), str):
            shared_plan_sha256 = str(shared_value.item())
        else:
            shared_plan_sha256 = np.asarray(
                shared_value, dtype=np.uint8
            ).tobytes().hex()
        if len(shared_plan_sha256) != 64:
            raise ReplayDatasetBuildError(
                "joint fake target event-plan digest is malformed"
            )
        joint_zero_fields = (
            "physical_out_of_bounds_point_count",
            "physical_out_of_bounds_point_rate",
            "physical_clip_count",
            "physical_clip_rate",
            "physical_clipped_coordinate_value_count",
            "target_clipping_applied",
            "slot_dropped",
        )
        if any(float(np.asarray(archive[name]).item()) != 0.0 for name in joint_zero_fields):
            raise ReplayDatasetBuildError("joint fake target failed no-clip/OOB gate")
        if not all(
            bool(np.asarray(archive[name]).item())
            for name in (
                "typed_target_dispatch_feasibility_gate_pass",
                "generated_dispatch_quality_gate_pass",
            )
        ):
            raise ReplayDatasetBuildError("joint fake target dispatch gate failed")
    if (
        observed_action != action
        or not trajectory_source.is_file()
        or len(trajectory_source_sha256) != 64
    ):
        raise ReplayDatasetBuildError(f"{joint_path}: fake action/source mismatch")
    values = _load_trajectory_archive(trajectory_source, trajectory_cache)
    actual_source_sha256 = str(np.asarray(values["__source_sha256"]).item())
    if trajectory_source_sha256 != actual_source_sha256:
        raise ReplayDatasetBuildError(
            f"{joint_path}: fake trajectory source hash mismatch"
        )
    count = len(values["duration_ms"])
    offsets = np.asarray(values["android_offsets"], dtype=np.int64)
    if not 0 <= archive_index < count or offsets.shape != (count + 1,):
        raise ReplayDatasetBuildError("fake trajectory archive index is invalid")
    if (
        str(np.asarray(values["action"]).item()) != action
        or int(values["orientation_id"][archive_index]) != orientation
    ):
        raise ReplayDatasetBuildError("fake trajectory archive identity changed")
    source_user = int(np.asarray(values["user_id"])[archive_index])
    source_split = int(np.asarray(values["split_id"])[archive_index])
    source_sample = int(np.asarray(values["sample_index"])[archive_index])
    plan_bytes = np.asarray(
        values["event_plan_sha256"][archive_index], dtype=np.uint8
    )
    if plan_bytes.shape != (32,):
        raise ReplayDatasetBuildError("fake target event-plan digest is malformed")
    plan_sha256 = plan_bytes.tobytes().hex()
    source_zero_fields = (
        "clipped_point_count",
        "clipped_point_rate",
        "geometry_outlier",
        "pre_projection_oob_point_count",
        "pre_projection_oob_point_rate",
        "target_clipping_applied",
        "slot_dropped",
        "physical_clip_count",
        "physical_clip_rate",
        "physical_clipped_coordinate_value_count",
    )
    if any(
        float(np.asarray(values[name])[archive_index]) != 0.0
        for name in source_zero_fields
    ):
        raise ReplayDatasetBuildError("fake target failed no-clip/OOB geometry gate")
    if (
        not bool(np.asarray(values["geometry_valid"])[archive_index])
        or not bool(
            np.asarray(values["typed_target_dispatch_feasibility_gate_pass"])[
                archive_index
            ]
        )
        or not bool(
            np.asarray(values["generated_dispatch_quality_gate_pass"])[
                archive_index
            ]
        )
    ):
        raise ReplayDatasetBuildError("fake target geometry/dispatch gate failed")
    exclusion = str(
        np.asarray(values["geometry_exclusion_code"])[archive_index]
    )
    if exclusion not in {"", "none", "None", "0"}:
        raise ReplayDatasetBuildError("fake target has a geometry exclusion code")
    if expected_user_id is not None:
        prefix = "hmog_u"
        if not str(expected_user_id).startswith(prefix):
            raise ReplayDatasetBuildError("fake target expected user ID is malformed")
        expected_numeric_user = int(str(expected_user_id)[len(prefix) :])
        if source_user != expected_numeric_user:
            raise ReplayDatasetBuildError("fake target user binding changed")
    split_ids = {"train": 0, "development": 1, "test": 2}
    if expected_split is not None and source_split != split_ids.get(expected_split):
        raise ReplayDatasetBuildError("fake target split binding changed")
    if (
        expected_sample_idx is not None
        and (
            source_sample != int(expected_sample_idx)
            or paired_sample_index != int(expected_sample_idx)
        )
    ):
        raise ReplayDatasetBuildError("fake target sample binding changed")
    if (
        expected_cross_modal_pair_id is not None
        and (
            plan_sha256 != str(expected_cross_modal_pair_id)
            or shared_plan_sha256 != str(expected_cross_modal_pair_id)
        )
    ):
        raise ReplayDatasetBuildError("fake target cross-modal binding changed")
    left, right = (int(value) for value in offsets[archive_index : archive_index + 2])
    if right - left < 2:
        raise ReplayDatasetBuildError("fake target has too few Android rows")
    x = np.asarray(values["flat_android_x"][left:right], dtype=np.float64)
    y = np.asarray(values["flat_android_y"][left:right], dtype=np.float64)
    t_ms = np.asarray(
        values["flat_android_t_ms"][left:right], dtype=np.float64
    )
    pressure = np.asarray(
        values["flat_android_pressure"][left:right], dtype=np.float64
    )
    pointers = np.asarray(
        values["flat_android_pointer_id"][left:right], dtype=np.int64
    )
    actions = np.asarray(values["flat_android_action"][left:right], dtype=np.int64)
    frames = np.asarray(
        values["flat_android_frame_index"][left:right], dtype=np.int64
    )
    frame_end = np.asarray(
        values["flat_android_frame_end"][left:right], dtype=np.uint8
    )
    key_indices = np.asarray(
        values["flat_android_key_index"][left:right], dtype=np.int64
    )
    raw_duration = float(np.asarray(values["duration_ms"])[archive_index])
    if (
        not np.isfinite(raw_duration)
        or raw_duration <= 0.0
        or not np.isfinite(t_ms).all()
        or np.any(np.diff(t_ms) < 0.0)
        or float(t_ms[-1] - t_ms[0]) > raw_duration + 1.0e-3
    ):
        raise ReplayDatasetBuildError("fake target Android timing is invalid")
    direction: str | None = None
    pinch_scale: str | None = None
    keycodes: tuple[int, ...] = ()
    if action in {"scroll", "swipe", "pinch"}:
        request = classify_replay_request(
            action=action,
            orientation_id=orientation,
            target_duration_ms=target_duration_ms,
            x_px=x,
            y_px=y,
            pointer_id=pointers if action == "pinch" else None,
            android_action=actions if action == "pinch" else None,
            frame_index=frames if action == "pinch" else None,
        )
        direction = request.direction
        pinch_scale = request.pinch_scale_direction
    elif action == "keystroke":
        keys = key_indices
        codes = np.asarray(
            values["flat_android_keycode"][left:right], dtype=np.int64
        )
        ordered: list[int] = []
        for key in keys:
            if key < 0 or int(key) in ordered:
                continue
            ordered.append(int(key))
        extracted: list[int] = []
        for key in ordered:
            observed = np.unique(codes[keys == key])
            if len(observed) != 1:
                raise ReplayDatasetBuildError("fake key has inconsistent keycodes")
            extracted.append(int(observed[0]))
        if not extracted:
            raise ReplayDatasetBuildError("fake keystroke target has no keycodes")
        keycodes = tuple(extracted)
    return AndroidTarget(
        action=action,
        orientation_id=orientation,
        direction=direction,
        pinch_scale_direction=pinch_scale,
        keycodes=keycodes,
        trajectory_source=trajectory_source,
        trajectory_source_sha256=trajectory_source_sha256,
        trajectory_archive_index=archive_index,
        raw_duration_ms=raw_duration,
        t_ms=t_ms.copy(),
        x_px=x.copy(),
        y_px=y.copy(),
        pressure=pressure.copy(),
        pointer_id=pointers.copy(),
        android_action=actions.copy(),
        key_index=key_indices.copy(),
        frame_index=frames.copy(),
        frame_end=frame_end.copy(),
        bound_event_plan_sha256=plan_sha256,
    )


def observe_bound_android_target(
    target: AndroidTarget,
    *,
    target_samples: int,
    target_duration_ms: float,
) -> TouchObservation:
    """Apply the common observer to an existing fake target's Android rows."""

    return observe_android_rows(
        action=target.action,
        target_samples=target_samples,
        orientation_id=target.orientation_id,
        t_ms=target.t_ms,
        x_px=target.x_px,
        y_px=target.y_px,
        pressure=target.pressure,
        pointer_id=target.pointer_id,
        android_action=target.android_action,
        frame_index=target.frame_index,
        frame_end=target.frame_end,
        key_index=target.key_index,
        source_duration_ms=target.raw_duration_ms,
        target_duration_ms=target_duration_ms,
    )


def _observe_conditional_touch_target(
    generator: ConditionalTouchGenerator,
    target: AndroidTarget,
    *,
    target_samples: int,
    target_duration_ms: float,
    generator_seed: int,
    requested_start_xy: Sequence[float] | None = None,
    requested_end_xy: Sequence[float] | None = None,
    requested_direction: str | None = None,
) -> tuple[TouchObservation, dict[str, Any]]:
    """Generate one smoke touch from explicit target geometry and timing."""

    if target.action not in SINGLE_POINTER_TOUCH_ACTIONS:
        raise ReplayDatasetBuildError(
            "conditional touch generation supports only tap/scroll/swipe"
        )
    if len(target.t_ms) < 2:
        raise ReplayDatasetBuildError(
            "conditional touch target has fewer than two raw timestamps"
        )
    lifecycle_start, lifecycle_end = _single_pointer_target_control_points_px(
        target
    )
    base_actions = np.asarray(target.android_action, dtype=np.int64) & ACTION_MASK
    if (
        int(base_actions[0]) != ACTION_DOWN
        or int(base_actions[-1]) != ACTION_UP
    ):
        raise ReplayDatasetBuildError(
            "conditional target lifecycle must begin at DOWN and end at UP"
        )
    explicit_geometry = (
        requested_start_xy is not None or requested_end_xy is not None
    )
    if explicit_geometry and (
        requested_start_xy is None or requested_end_xy is None
    ):
        raise ReplayDatasetBuildError(
            "explicit conditional touch request requires both start and end"
        )
    if not explicit_geometry and requested_direction is not None:
        raise ReplayDatasetBuildError(
            "explicit conditional touch direction requires start and end"
        )

    width_px, height_px = screen_dimensions_for_orientation(
        target.orientation_id
    )

    def request_xy(name: str, value: Sequence[float]) -> tuple[float, float]:
        point = np.asarray(value, dtype=np.float64)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ReplayDatasetBuildError(
                f"conditional touch {name} must contain two finite coordinates"
            )
        if not (
            0.0 <= point[0] <= width_px
            and 0.0 <= point[1] <= height_px
        ):
            raise ReplayDatasetBuildError(
                f"conditional touch {name} leaves the physical screen"
            )
        return (float(point[0]), float(point[1]))

    if explicit_geometry:
        start_xy = request_xy("requested_start_xy", requested_start_xy)
        end_xy = request_xy("requested_end_xy", requested_end_xy)
        direction = requested_direction
    else:
        start_xy = lifecycle_start
        # The bound DOWN/UP pair is the explicit request.  Earlier smoke code
        # silently collapsed every tap to end=start; that both violated the
        # endpoint contract and removed genuine micro-movement.  Keep the
        # supplied endpoint for every single-pointer action.
        end_xy = lifecycle_end
        direction = None if target.action == "tap" else target.direction

    start_array = np.asarray(start_xy, dtype=np.float64)
    end_array = np.asarray(end_xy, dtype=np.float64)
    delta = end_array - start_array
    endpoint_equal = bool(np.array_equal(start_array, end_array))
    if endpoint_equal:
        expected_direction = "stationary"
    else:
        direction_labels = (
            "right",
            "down_right",
            "down",
            "down_left",
            "left",
            "up_left",
            "up",
            "up_right",
        )
        angle = float(np.arctan2(delta[1], delta[0]))
        direction_index = int(
            np.floor((angle + np.pi / 8.0) / (np.pi / 4.0))
        ) % len(direction_labels)
        expected_direction = direction_labels[direction_index]
    if target.action == "tap":
        allowed_directions = (
            (None, "stationary")
            if endpoint_equal
            else (None, expected_direction)
        )
        if direction not in allowed_directions:
            raise ReplayDatasetBuildError(
                "conditional tap direction does not match requested endpoints"
            )
    elif direction is None:
        raise ReplayDatasetBuildError(
            "conditional scroll/swipe target lacks a direction"
        )
    elif direction != expected_direction:
        raise ReplayDatasetBuildError(
            "conditional scroll/swipe direction does not match requested endpoints"
        )
    try:
        generated = generator.generate(
            action=target.action,
            orientation_id=target.orientation_id,
            start_xy_px=start_xy,
            end_xy_px=end_xy,
            direction=direction,
            seed=int(generator_seed),
            t_ms=target.t_ms,
            detector_sample_count=int(target_samples),
        )
    except ConditionalTouchGeneratorError as exc:
        raise ReplayDatasetBuildError(
            f"conditional touch generation failed: {exc}"
        ) from exc

    key_index = np.full(len(generated.t_ms), -1, dtype=np.int64)
    observation = observe_android_rows(
        action=target.action,
        target_samples=int(target_samples),
        orientation_id=target.orientation_id,
        t_ms=generated.t_ms,
        x_px=generated.x_px,
        y_px=generated.y_px,
        pressure=generated.pressure,
        pointer_id=generated.pointer_id,
        android_action=generated.android_action,
        frame_index=generated.frame_index,
        frame_end=generated.frame_end,
        key_index=key_index,
        source_duration_ms=target.raw_duration_ms,
        target_duration_ms=float(target_duration_ms),
    )

    requested = np.asarray((start_xy, end_xy), dtype=np.float64)
    raw_output = np.asarray(
        (
            (generated.x_px[0], generated.y_px[0]),
            (generated.x_px[-1], generated.y_px[-1]),
        ),
        dtype=np.float64,
    )
    detector_output_rel = np.asarray(
        (
            observation.trajectory[0, 1:3],
            observation.trajectory[-1, 1:3],
        ),
        dtype=np.float64,
    )
    detector_output = detector_output_rel * np.asarray(
        (width_px, height_px), dtype=np.float64
    )
    raw_errors = np.linalg.norm(raw_output - requested, axis=1)
    detector_errors = np.linalg.norm(detector_output - requested, axis=1)
    maximum_error = float(max(np.max(raw_errors), np.max(detector_errors)))
    # Raw endpoints remain bit-exact.  Detector trajectories store normalized
    # coordinates as float32, so allow only the documented sub-millipixel
    # round-trip tolerance at this representation boundary.
    if maximum_error > 5.0e-4:
        raise ReplayDatasetBuildError(
            "conditional touch output did not preserve exact start/end geometry"
        )
    if generated.realized_direction != expected_direction:
        raise ReplayDatasetBuildError(
            "conditional touch output direction differs from its request"
        )
    if target.action == "tap" and (
        bool(generated.tap_stationary_branch) != endpoint_equal
    ):
        raise ReplayDatasetBuildError(
            "conditional tap output selected the wrong endpoint branch"
        )
    generated_duration = float(generated.t_ms[-1] - generated.t_ms[0])
    requested_duration = float(target.t_ms[-1] - target.t_ms[0])
    if not math.isclose(
        generated_duration, requested_duration, rel_tol=0.0, abs_tol=1.0e-9
    ):
        raise ReplayDatasetBuildError(
            "conditional touch output duration differs from its raw request"
        )
    return observation, {
        "generator_seed": int(generator_seed),
        "conditioning_action": target.action,
        "conditioning_orientation_id": int(target.orientation_id),
        "conditioning_direction": direction,
        "requested_start_px": requested[0].tolist(),
        "requested_end_px": requested[1].tolist(),
        "raw_output_start_px": raw_output[0].tolist(),
        "raw_output_end_px": raw_output[1].tolist(),
        "detector_output_start_px": detector_output[0].tolist(),
        "detector_output_end_px": detector_output[1].tolist(),
        "raw_start_error_px": float(raw_errors[0]),
        "raw_end_error_px": float(raw_errors[1]),
        "detector_start_error_px": float(detector_errors[0]),
        "detector_end_error_px": float(detector_errors[1]),
        "maximum_endpoint_error_px": maximum_error,
        "requested_raw_duration_ms": requested_duration,
        "generated_raw_duration_ms": generated_duration,
        "requested_raw_row_count": int(len(target.t_ms)),
        "generated_raw_row_count": int(len(generated.t_ms)),
        "residual_scale": float(generated.residual_scale),
        "tap_stationary_branch": bool(generated.tap_stationary_branch),
        "realized_direction": generated.realized_direction,
        "generation_mode": "frozen_conditional_touch_generator",
    }


def _resample_touch_template(
    template: np.ndarray, target_samples: int
) -> np.ndarray:
    """Put a donor template on the carrier's sample grid."""

    values = np.asarray(template, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 9:
        raise ReplayDatasetBuildError("exact touch template must have 9 columns")
    if len(values) < 2:
        raise ReplayDatasetBuildError("exact touch template needs two samples")
    target = int(target_samples)
    if target < 2:
        raise ReplayDatasetBuildError("exact touch carrier needs two samples")
    if len(values) == target:
        return values

    source_positions = np.linspace(0.0, 1.0, len(values), dtype=np.float64)
    target_positions = np.linspace(0.0, 1.0, target, dtype=np.float64)
    resampled = np.empty((target, 9), dtype=np.float32)
    resampled[:, 7] = np.interp(
        target_positions,
        source_positions,
        values[:, 7].astype(np.float64),
    ).astype(np.float32)
    held = np.clip(
        np.floor(target_positions * (len(values) - 1) + 0.5).astype(np.int64),
        0,
        len(values) - 1,
    )
    for column in (0, 1, 2, 3, 4, 8):
        resampled[:, column] = values[held, column]
    contact = resampled[:, 0]
    deltas = np.zeros((target, 2), dtype=np.float32)
    consecutive = (contact[1:] > 0.0) & (contact[:-1] > 0.0)
    differences = np.diff(resampled[:, 1:3], axis=0)
    deltas[1:][consecutive] = differences[consecutive]
    resampled[:, 5:7] = deltas
    return resampled


def _pinch_material_widest_span(material: Any) -> float:
    """Report the widest two-pointer span one frozen pinch ever reaches."""

    return max(
        float(material.row["pinch_start_span_px"]),
        float(material.row["pinch_end_span_px"]),
    )


def _pinch_requested_widest_span(requested: PinchEndpointGeometry) -> float:
    """Report the extent a requested pinch area has to accommodate."""

    return max(float(requested.start_span_px), float(requested.end_span_px))


def _fiveshot_timing_samples(duration_ms: float) -> int:
    """Convert a duration into the row count the detector grid reports it on."""

    if not math.isfinite(duration_ms) or duration_ms <= 0.0:
        raise ReplayDatasetBuildError("a gesture duration must be positive")
    return int(round(float(duration_ms) / FIVESHOT_TIMING_PERIOD_MS)) + 1


def _carrier_binding_path(trajectory_source: Path) -> Path:
    """Locate the crosswalk that says which cached IMU a carrier was built on."""

    parts = list(Path(trajectory_source).resolve().parts)
    try:
        index = len(parts) - 1 - parts[::-1].index("shards")
    except ValueError as error:
        raise ReplayDatasetBuildError(
            f"carrier archive is not under a shards root: {trajectory_source}"
        ) from error
    parts[index] = "analytic_bindings"
    return Path(*parts).with_suffix(".jsonl")


@functools.lru_cache(maxsize=64)
def _carrier_imu_sources(trajectory_source: str) -> tuple[str, ...]:
    """Read the cached IMU path behind every carrier in one archive."""

    path = _carrier_binding_path(Path(trajectory_source))
    if not path.is_file():
        raise ReplayDatasetBuildError(f"carrier bindings are missing: {path}")
    sources: dict[int, str] = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            sources[int(row["output_archive_index"])] = str(row["imu_source"])
    if not sources or sorted(sources) != list(range(len(sources))):
        raise ReplayDatasetBuildError(
            f"carrier bindings do not index their archive contiguously: {path}"
        )
    return tuple(sources[index] for index in range(len(sources)))


@functools.lru_cache(maxsize=4096)
def _carrier_imu_window(imu_source: str) -> tuple[np.ndarray, np.ndarray]:
    """Read one carrier's padded IMU window and the span its action occupies."""

    path = Path(imu_source)
    if not path.is_file():
        raise ReplayDatasetBuildError(f"carrier IMU cache is missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if "window" not in archive.files or "mask" not in archive.files:
            raise ReplayDatasetBuildError(
                f"carrier IMU cache carries no padded window: {path}"
            )
        window = np.ascontiguousarray(archive["window"], dtype=np.float32)
        mask = np.ascontiguousarray(archive["mask"]).astype(bool)
    window.flags.writeable = False
    mask.flags.writeable = False
    return window, mask


def _match_keystroke_imu_samples(
    imu: np.ndarray, target_samples: int
) -> np.ndarray:
    """Put a synthesised typing IMU on its carrier's sample grid."""

    values = np.asarray(imu, dtype=np.float32)
    target = int(target_samples)
    if values.ndim != 2 or values.shape[1] != IMU_CHANNELS or len(values) < 2:
        raise ReplayDatasetBuildError("synthesised keystroke IMU is malformed")
    if target < 2:
        raise ReplayDatasetBuildError("keystroke carrier needs two samples")
    if len(values) == target:
        return values
    source_positions = np.linspace(0.0, 1.0, len(values), dtype=np.float64)
    target_positions = np.linspace(0.0, 1.0, target, dtype=np.float64)
    resampled = np.empty((target, values.shape[1]), dtype=np.float32)
    for channel in range(values.shape[1]):
        resampled[:, channel] = np.interp(
            target_positions, source_positions, values[:, channel].astype(np.float64)
        ).astype(np.float32)
    return resampled


def _wrap_pinch_angle(angle: float) -> float:
    """Report a similarity rotation in (-pi, pi] so the audit stays readable."""

    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)


def _observe_pinch_area_similarity(
    donor: Any,
    target: AndroidTarget,
    *,
    target_samples: int,
    target_duration_ms: float,
    requested: PinchEndpointGeometry,
    pointer_order: str = "as_requested",
) -> tuple[TouchObservation, dict[str, Any]]:
    """Place one frozen real pinch on the requested on-screen area."""

    if target.action != "pinch":
        raise ReplayDatasetBuildError(
            "pinch area similarity requires a pinch Android target"
        )
    values = _resample_touch_template(
        np.asarray(donor.trajectory, dtype=np.float32), int(target_samples)
    )
    if values.shape != (int(target_samples), 9) or not np.isfinite(values).all():
        raise ReplayDatasetBuildError("pinch material is malformed for its carrier")
    if (
        np.any(values[:, 1:3] < 0.0)
        or np.any(values[:, 1:3] > 1.0)
        or np.any(np.diff(values[:, 7]) < 0.0)
    ):
        raise ReplayDatasetBuildError(
            "pinch material violates screen or time invariants"
        )

    if pointer_order not in PINCH_POINTER_ORDERS:
        raise ReplayDatasetBuildError(f"unknown pinch pointer order {pointer_order!r}")
    row = donor.row
    try:
        source_start = np.asarray(row["pinch_start_points_px"], dtype=np.float64)
        source_end = np.asarray(row["pinch_end_points_px"], dtype=np.float64)
        source_start_span = float(row["pinch_start_span_px"])
        source_end_span = float(row["pinch_end_span_px"])
        source_direction = str(row["pinch_scale_direction"])
        source_orientation = int(row["orientation_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ReplayDatasetBuildError(
            "pinch material lacks frozen live two-pointer geometry"
        ) from exc
    if (
        source_start.shape != (2, 2)
        or source_end.shape != (2, 2)
        or not np.isfinite(source_start).all()
        or not np.isfinite(source_end).all()
        or not np.isfinite(source_start_span)
        or not np.isfinite(source_end_span)
        or min(source_start_span, source_end_span) <= 0.0
    ):
        raise ReplayDatasetBuildError("pinch material geometry is not usable")
    source_percent = source_end_span / source_start_span

    widest_is_end = source_end_span > source_start_span
    source_widest = source_end if widest_is_end else source_start
    source_widest_span = source_end_span if widest_is_end else source_start_span
    source_widest_center = source_widest.mean(axis=0)
    source_widest_vector = source_widest[1] - source_widest[0]

    requested_widest_is_end = float(requested.end_span_px) > float(
        requested.start_span_px
    )
    requested_widest = np.asarray(
        requested.end_points_px if requested_widest_is_end
        else requested.start_points_px,
        dtype=np.float64,
    )
    requested_widest_span = float(
        requested.end_span_px if requested_widest_is_end
        else requested.start_span_px
    )
    if not np.isfinite(requested_widest_span) or requested_widest_span <= 0.0:
        raise ReplayDatasetBuildError("pinch request has no usable area span")
    requested_widest_center = requested_widest.mean(axis=0)
    requested_widest_vector = requested_widest[1] - requested_widest[0]
    if pointer_order == "swapped":
        requested_widest_vector = -requested_widest_vector

    scale = requested_widest_span / source_widest_span
    rotation = float(
        np.arctan2(requested_widest_vector[1], requested_widest_vector[0])
        - np.arctan2(source_widest_vector[1], source_widest_vector[0])
    )
    cosine, sine = float(np.cos(rotation)), float(np.sin(rotation))
    matrix = scale * np.asarray(((cosine, -sine), (sine, cosine)), dtype=np.float64)

    def _place(points: np.ndarray) -> np.ndarray:
        return (
            np.atleast_2d(np.asarray(points, dtype=np.float64))
            - source_widest_center[None, :]
        ) @ matrix.T + requested_widest_center[None, :]

    width_px, height_px = screen_dimensions_for_orientation(target.orientation_id)
    screen = np.asarray((width_px, height_px), dtype=np.float64)
    # The recording's own screen may differ from the carrier's, so its stored
    # relative coordinates are de-normalised with its own orientation first.
    source_width, source_height = screen_dimensions_for_orientation(
        source_orientation
    )
    source_xy = values[:, 1:3].astype(np.float64) * np.asarray(
        (source_width, source_height), dtype=np.float64
    )[None, :]
    moved = _place(source_xy)
    # Both fingers have to be dispatchable, not just their centroid: every fake
    # carrier in this release reports zero out-of-bounds points and zero clips,
    # so a placed pinch that puts a finger off the screen is not admissible.
    placed_fingers = np.vstack((_place(source_start), _place(source_end)))

    def _outside(points: np.ndarray) -> int:
        return int(
            np.count_nonzero(
                (points[:, 0] < 0.0)
                | (points[:, 0] > float(width_px))
                | (points[:, 1] < 0.0)
                | (points[:, 1] > float(height_px))
            )
        )

    off_screen = _outside(moved)
    off_screen_fingers = _outside(placed_fingers)
    if off_screen or off_screen_fingers:
        raise ReplayDatasetBuildError(
            f"placed pinch leaves the screen at {off_screen} centroid samples "
            f"and {off_screen_fingers} finger endpoints"
        )

    trajectory = values.copy()
    trajectory[:, 1] = (moved[:, 0] / float(width_px)).astype(np.float32)
    trajectory[:, 2] = (moved[:, 1] / float(height_px)).astype(np.float32)
    elapsed = values[:, 7].astype(np.float64) - float(values[0, 7])
    span = float(elapsed[-1])
    if not np.isfinite(span) or span <= 0.0:
        raise ReplayDatasetBuildError("pinch material has no elapsed timeline")
    # The clock column is the observer's regular grid.  Rescaling the donor's
    # own clock reaches the same nominal seconds by different arithmetic and
    # leaves a float32 rounding signature a detector can separate on.
    trajectory[:, 7] = detector_grid_clock(
        len(trajectory), float(target_duration_ms)
    )
    deltas = np.zeros((len(trajectory), 2), dtype=np.float32)
    consecutive = (trajectory[1:, 0] > 0.0) & (trajectory[:-1, 0] > 0.0)
    differences = np.diff(trajectory[:, 1:3], axis=0)
    deltas[1:][consecutive] = differences[consecutive]
    trajectory[:, 5:7] = deltas

    delivered_widest = _place(source_widest)
    if pointer_order == "swapped":
        delivered_widest = delivered_widest[::-1]
    widest_error = float(
        np.max(np.linalg.norm(delivered_widest - requested_widest, axis=1))
    )
    if widest_error > 5.0e-4:
        raise ReplayDatasetBuildError(
            "placed pinch did not land on the requested area extent"
        )
    generated_duration = float((trajectory[-1, 7] - trajectory[0, 7]) * 1000.0)
    if not math.isclose(
        generated_duration, float(target_duration_ms), rel_tol=0.0, abs_tol=1.0e-3
    ):
        raise ReplayDatasetBuildError(
            "placed pinch did not preserve the carrier duration"
        )
    observation = TouchObservation(
        touch=trajectory[:, :7].copy(),
        trajectory=trajectory.copy(),
        touch_observed=bool(np.any(trajectory[:, 8] > 0.5)),
        source_updates=int(len(trajectory)),
    )
    return observation, {
        "conditioning_action": "pinch",
        "conditioning_orientation_id": int(target.orientation_id),
        "requested_area_points_px": requested_widest.tolist(),
        "requested_area_center_px": requested_widest_center.tolist(),
        "requested_area_span_px": requested_widest_span,
        "requested_area_axis_rad": float(
            np.arctan2(requested_widest_vector[1], requested_widest_vector[0])
        ),
        "requested_area_moment": "end" if requested_widest_is_end else "start",
        "requested_carrier_scale_direction": requested.scale_direction,
        "pinch_pointer_order": str(pointer_order),
        "delivered_area_points_px": delivered_widest.tolist(),
        "delivered_area_extent_error_px": widest_error,
        "similarity_scale": float(scale),
        "similarity_rotation_rad": float(_wrap_pinch_angle(rotation)),
        "source_widest_span_px": source_widest_span,
        "source_widest_axis_rad": float(
            np.arctan2(source_widest_vector[1], source_widest_vector[0])
        ),
        "source_widest_moment": "end" if widest_is_end else "start",
        "source_start_span_px": source_start_span,
        "source_end_span_px": source_end_span,
        "source_percent": source_percent,
        "source_scale_direction": source_direction,
        "source_orientation_id": source_orientation,
        "differential_deformation_used": False,
        "off_screen_sample_count": 0,
        "off_screen_finger_endpoint_count": 0,
        "requested_raw_duration_ms": float(target_duration_ms),
        "generated_raw_duration_ms": generated_duration,
        "requested_raw_row_count": int(len(trajectory)),
        "generated_raw_row_count": int(len(trajectory)),
        "generation_mode": PINCH_AREA_SIMILARITY_GENERATION_MODE,
    }


def _fiveshot_tap_drift_request(
    template: np.ndarray,
    target: AndroidTarget,
    *,
    start_px: Sequence[float],
) -> tuple[np.ndarray, tuple[float, float], str | None, dict[str, Any]]:
    """Request a tap that keeps its donor's own DOWN-to-UP drift."""

    values = np.asarray(template, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 9 or len(values) < 2:
        raise ReplayDatasetBuildError("five-shot tap donor template is malformed")
    width_px, height_px = screen_dimensions_for_orientation(
        target.orientation_id
    )
    dimensions = np.asarray((width_px, height_px), dtype=np.float64)
    start = np.asarray(start_px, dtype=np.float64)
    if start.shape != (2,) or not np.isfinite(start).all():
        raise ReplayDatasetBuildError("five-shot tap request start is invalid")
    coordinates = values[:, 1:3].astype(np.float64)
    offsets = (coordinates - coordinates[0][None, :]) * dimensions[None, :]
    donor_drift_px = float(np.linalg.norm(offsets[-1]))

    # Aim a shrunk request just inside the budget: the scaled coordinates ride
    # back through float32 storage, so landing exactly on the limit would let
    # the recovered chord round past it.
    drift_budget_px = (
        FIVESHOT_TAP_DRIFT_LIMIT_PX - FIVESHOT_TAP_SCREEN_MARGIN_PX
    )
    scale = 1.0
    if donor_drift_px > drift_budget_px:
        scale = drift_budget_px / donor_drift_px
    # Translating the donor onto the target anchor can push a sample off the
    # physical screen; one common factor keeps every sample inside it.  The
    # shrunk coordinates ride back through float32 storage, whose quantum is
    # about 1e-4 px on a 1920 px axis, so the room granted here stops that short
    # of the boundary rather than exactly on it.
    for axis in range(2):
        delta = offsets[:, axis]
        positive = delta > 0.0
        negative = delta < 0.0
        if np.any(positive):
            room = max(
                0.0,
                float(dimensions[axis] - start[axis])
                - FIVESHOT_TAP_SCREEN_MARGIN_PX,
            )
            scale = min(scale, float(np.min(room / delta[positive])))
        if np.any(negative):
            room = max(
                0.0, float(start[axis]) - FIVESHOT_TAP_SCREEN_MARGIN_PX
            )
            scale = min(scale, float(np.min(room / -delta[negative])))
    scale = float(max(0.0, min(1.0, scale)))

    request_template = values
    if scale < 1.0:
        request_template = values.copy()
        request_template[:, 1:3] = (
            coordinates[0][None, :] + offsets * scale / dimensions[None, :]
        ).astype(np.float32)
        # dx/dy stay the held differences of the coordinates they describe.
        deltas = np.zeros((len(request_template), 2), dtype=np.float32)
        contact = request_template[:, 0]
        consecutive = (contact[1:] > 0.0) & (contact[:-1] > 0.0)
        differences = np.diff(request_template[:, 1:3], axis=0)
        deltas[1:][consecutive] = differences[consecutive]
        request_template[:, 5:7] = deltas

    # Read the chord back through the same float32 storage and float64 screen
    # arithmetic the generator uses, so the request it receives is the chord it
    # will recompute rather than an ideal value it can never reproduce.
    source_px = request_template[:, 1:3].astype(np.float64) * dimensions[None, :]
    chord = source_px[-1] - source_px[0]
    end = start + chord
    if np.any(end < 0.0) or np.any(end > dimensions):
        raise ReplayDatasetBuildError(
            "five-shot tap drift request left the physical screen"
        )
    requested_drift_px = float(np.linalg.norm(end - start))
    if requested_drift_px > FIVESHOT_TAP_DRIFT_LIMIT_PX:
        raise ReplayDatasetBuildError(
            "five-shot tap drift request exceeded the touch slop budget"
        )
    if np.array_equal(end, start):
        direction: str | None = None
    else:
        angle = float(np.arctan2(float(chord[1]), float(chord[0])))
        index = int(np.floor((angle + np.pi / 8.0) / (np.pi / 4.0))) % 8
        direction = EXACT_TOUCH_DIRECTION8[index]
    audit = {
        "tap_donor_drift_px": donor_drift_px,
        "tap_requested_drift_px": requested_drift_px,
        "tap_donor_drift_scale": scale,
        "tap_drift_limit_px": FIVESHOT_TAP_DRIFT_LIMIT_PX,
    }
    return request_template, (float(end[0]), float(end[1])), direction, audit


def _donor_report_clock(
    times_ms: np.ndarray,
) -> tuple[float, np.ndarray, float]:
    """Recover the device report cadence the donor was recorded at."""

    gaps = np.diff(np.asarray(times_ms, dtype=np.float64))
    if len(gaps) < 2:
        return 0.0, np.ones(1, dtype=np.int64), 0.0
    head = float(gaps[0])
    body = gaps[1:]
    body = body[body > 0.0]
    if not len(body):
        return 0.0, np.ones(1, dtype=np.int64), head
    period = float(np.median(body))
    for _ in range(6):
        if period <= 0.0:
            break
        ticks = np.maximum(np.round(body / period), 1.0)
        period = float(np.sum(body) / np.sum(ticks))
    if not np.isfinite(period) or period <= 0.0:
        return 0.0, np.ones(1, dtype=np.int64), head
    ticks = np.maximum(np.round(body / period), 1.0).astype(np.int64)
    return period, ticks, head


def _synthesised_update_times(
    period_ms: float,
    ticks: np.ndarray,
    head_ms: float,
    duration_ms: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Emit updates at the victim's own cadence across the carrier's window."""

    duration = float(duration_ms)
    if period_ms <= 0.0 or duration <= 0.0:
        return np.asarray((0.0, duration), dtype=np.float64)
    head = min(max(float(head_ms), 0.0), duration * 0.5)
    times = [0.0]
    if head > 0.0:
        times.append(head)
    while True:
        step = float(period_ms) * float(ticks[int(rng.integers(len(ticks)))])
        nxt = times[-1] + step
        if nxt >= duration - 1.0:
            break
        times.append(nxt)
    times.append(duration)
    return np.asarray(times, dtype=np.float64)


def _observe_synthesised_clock_touch(
    template: np.ndarray,
    target: AndroidTarget,
    *,
    target_samples: int,
    target_duration_ms: float,
    requested_start_xy: Sequence[float],
    requested_end_xy: Sequence[float],
    requested_direction: str | None,
    seed: int,
) -> tuple[TouchObservation, dict[str, Any]]:
    """Inject the donor's own vertices on the victim's own cadence, observe once."""

    values = np.asarray(template, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 9 or len(values) < 2:
        raise ReplayDatasetBuildError("synthesised-clock donor is malformed")
    width_px, height_px = screen_dimensions_for_orientation(
        target.orientation_id
    )
    coordinates = values[:, 1:3].astype(np.float64)
    moved = np.any(np.diff(coordinates, axis=0) != 0.0, axis=1)
    vertex = np.flatnonzero(np.concatenate(([True], moved)))
    if len(vertex) < 2:
        raise ReplayDatasetBuildError("synthesised-clock donor never moves")
    vertex_ms = (
        values[vertex, 7].astype(np.float64) - float(values[vertex[0], 7])
    ) * 1000.0
    period, ticks, head = _donor_report_clock(vertex_ms)
    rng = np.random.default_rng(_event_seed(str(target.bound_event_plan_sha256), seed=seed))
    times = _synthesised_update_times(
        period, ticks, head, float(target_duration_ms), rng
    )
    count = len(times)
    if count < 2:
        raise ReplayDatasetBuildError("synthesised clock produced no event")

    # Body rows carry the donor's own recorded points; only a window longer than
    # the recording has anything to interpolate, and then the whole body is
    # interpolated rather than mixing invented points among real ones.
    body = max(count - 2, 0)
    points = np.empty((count, 2), dtype=np.float64)
    pressure = np.empty(count, dtype=np.float64)
    points[0] = coordinates[vertex[0]]
    points[-1] = coordinates[vertex[-1]]
    pressure[0] = float(values[vertex[0], 3])
    pressure[-1] = float(values[vertex[-1], 3])
    if body:
        span = float(vertex_ms[-1] - vertex_ms[0])
        donor_progress = (
            (vertex_ms - vertex_ms[0]) / span
            if span > 0.0
            else np.linspace(0.0, 1.0, len(vertex))
        )
        wanted = (times[1:-1] - times[0]) / max(float(times[-1] - times[0]), 1e-12)
        if body <= len(vertex):
            pick = np.maximum.accumulate(
                np.clip(
                    np.searchsorted(donor_progress, wanted, side="left"),
                    0,
                    len(vertex) - 1,
                )
            )
            points[1:-1] = coordinates[vertex[pick]]
            pressure[1:-1] = values[vertex[pick], 3].astype(np.float64)
        else:
            for axis in range(2):
                points[1:-1, axis] = np.interp(
                    wanted, donor_progress, coordinates[vertex, axis]
                )
            pressure[1:-1] = np.interp(
                wanted, donor_progress, values[vertex, 3].astype(np.float64)
            )

    direction = (
        EXACT_TOUCH_STATIONARY
        if requested_direction is None
        else str(requested_direction)
    )
    try:
        generated = generate_exact_touch_template(
            action=target.action,
            start_xy_px=requested_start_xy,
            end_xy_px=requested_end_xy,
            direction=direction,
            duration_ms=float(target_duration_ms),
            template_t_ms=times,
            template_x_px=points[:, 0] * float(width_px),
            template_y_px=points[:, 1] * float(height_px),
            template_pressure=pressure,
            screen_width_px=float(width_px),
            screen_height_px=float(height_px),
        )
    except ExactTouchTemplateError as exc:
        raise ReplayDatasetBuildError(
            f"synthesised-clock generation failed: {exc}"
        ) from exc

    lifecycle = np.full(count, ACTION_MOVE, dtype=np.int64)
    lifecycle[0] = ACTION_DOWN
    lifecycle[-1] = ACTION_UP
    try:
        observation = observe_android_rows(
            action=target.action,
            target_samples=int(target_samples),
            orientation_id=int(target.orientation_id),
            t_ms=generated.t_ms,
            x_px=generated.x_px,
            y_px=generated.y_px,
            pressure=generated.pressure,
            pointer_id=np.zeros(count, dtype=np.int64),
            android_action=lifecycle,
            frame_index=np.arange(count, dtype=np.int64),
            frame_end=np.ones(count, dtype=np.uint8),
            key_index=np.full(count, -1, dtype=np.int64),
            source_duration_ms=float(target_duration_ms),
            target_duration_ms=float(target_duration_ms),
        )
    except TouchObservationError as exc:
        raise ReplayDatasetBuildError(
            f"synthesised-clock observation failed: {exc}"
        ) from exc

    requested = np.asarray(
        (requested_start_xy, requested_end_xy), dtype=np.float64
    )
    detector_output = observation.trajectory[[0, -1], 1:3].astype(
        np.float64
    ) * np.asarray((width_px, height_px), dtype=np.float64)[None, :]
    detector_errors = np.linalg.norm(detector_output - requested, axis=1)
    maximum_error = float(np.max(detector_errors))
    if maximum_error > 5.0e-4:
        raise ReplayDatasetBuildError(
            "synthesised-clock output did not preserve requested endpoints"
        )
    delta = requested[1] - requested[0]
    stationary = bool(np.linalg.norm(delta) <= 1.0e-12)
    return observation, {
        "conditioning_action": target.action,
        "conditioning_orientation_id": int(target.orientation_id),
        "conditioning_direction": requested_direction,
        "requested_start_px": requested[0].tolist(),
        "requested_end_px": requested[1].tolist(),
        "raw_output_start_px": requested[0].tolist(),
        "raw_output_end_px": requested[1].tolist(),
        "detector_output_start_px": detector_output[0].tolist(),
        "detector_output_end_px": detector_output[1].tolist(),
        "raw_start_error_px": 0.0,
        "raw_end_error_px": 0.0,
        "detector_start_error_px": float(detector_errors[0]),
        "detector_end_error_px": float(detector_errors[1]),
        "maximum_endpoint_error_px": maximum_error,
        "requested_raw_duration_ms": float(target_duration_ms),
        "generated_raw_duration_ms": float(target_duration_ms),
        "requested_raw_row_count": int(count),
        "generated_raw_row_count": int(count),
        "residual_scale": float(generated.residual_scale),
        "transform_mode": generated.mode,
        "identity_transform": bool(generated.identity_transform),
        "tap_stationary_branch": bool(
            target.action == "tap" and stationary
        ),
        "realized_direction": (
            "stationary" if stationary else str(requested_direction)
        ),
        "generation_mode": "exact_touch_template_generator_v1",
        "synthesised_report_period_ms": float(period),
        "synthesised_update_count": int(count),
        "donor_update_count": int(len(vertex)),
    }


def _observe_exact_touch_template(
    template: np.ndarray,
    target: AndroidTarget,
    *,
    target_samples: int,
    target_duration_ms: float,
    requested_start_xy: Sequence[float],
    requested_end_xy: Sequence[float],
    requested_direction: str | None,
) -> tuple[TouchObservation, dict[str, Any]]:
    """Adapt a detector template to the new independent exact-touch API."""

    if target.action not in EXACT_TOUCH_TEMPLATE_ACTIONS:
        raise ReplayDatasetBuildError(
            "exact touch templates support only single-pointer touch actions"
        )
    values = _resample_touch_template(
        np.asarray(template, dtype=np.float32), int(target_samples)
    )
    if values.shape != (int(target_samples), 9):
        raise ReplayDatasetBuildError(
            "exact touch template sample count differs from its carrier"
        )
    if not np.isfinite(values).all():
        raise ReplayDatasetBuildError("exact touch template is non-finite")
    if (
        np.any(values[:, 1:3] < 0.0)
        or np.any(values[:, 1:3] > 1.0)
        or np.any(np.diff(values[:, 7]) < 0.0)
        or np.any(values[[0, -1], 0] <= 0.0)
        or np.any(values[[0, -1], 8] <= 0.0)
    ):
        raise ReplayDatasetBuildError(
            "exact touch template violates screen, time, or endpoint invariants"
        )
    expected_dx_dy = np.zeros((len(values), 2), dtype=np.float32)
    consecutive = (values[1:, 0] > 0.0) & (values[:-1, 0] > 0.0)
    differences = np.diff(values[:, 1:3], axis=0)
    expected_dx_dy[1:][consecutive] = differences[consecutive]
    if not np.array_equal(values[:, 5:7], expected_dx_dy):
        raise ReplayDatasetBuildError("exact touch template has invalid dx/dy")

    width_px, height_px = screen_dimensions_for_orientation(
        target.orientation_id
    )
    source_t_ms = (
        values[:, 7].astype(np.float64) - float(values[0, 7])
    ) * 1000.0
    direction = (
        EXACT_TOUCH_STATIONARY
        if requested_direction is None
        else str(requested_direction)
    )
    try:
        generated = generate_exact_touch_template(
            action=target.action,
            start_xy_px=requested_start_xy,
            end_xy_px=requested_end_xy,
            direction=direction,
            duration_ms=float(target_duration_ms),
            template_t_ms=source_t_ms,
            template_x_px=(
                values[:, 1].astype(np.float64) * float(width_px)
            ),
            template_y_px=(
                values[:, 2].astype(np.float64) * float(height_px)
            ),
            template_pressure=values[:, 3],
            screen_width_px=float(width_px),
            screen_height_px=float(height_px),
        )
    except ExactTouchTemplateError as exc:
        raise ReplayDatasetBuildError(
            f"exact touch template generation failed: {exc}"
        ) from exc

    # Identity requests retain the frozen detector bytes exactly.  Every other
    # request replaces only geometry, pressure, and elapsed time; lifecycle,
    # pointer identity, and availability remain coupled to the donor template.
    trajectory = values.copy()
    if not generated.identity_transform:
        trajectory[:, 1] = (
            generated.x_px / float(width_px)
        ).astype(np.float32)
        trajectory[:, 2] = (
            generated.y_px / float(height_px)
        ).astype(np.float32)
        trajectory[:, 3] = generated.pressure.astype(np.float32)
        # The generator's clock places the donor's samples in time; the column
        # written here is the observer's regular grid, and it has to be built by
        # the observer's own arithmetic.  Scaling the donor's clock instead
        # reaches the same nominal seconds with a different float32 rounding
        # signature, which separates real from fake without describing either.
        trajectory[:, 7] = detector_grid_clock(
            len(trajectory), float(target_duration_ms)
        )
        output_dx_dy = np.zeros((len(trajectory), 2), dtype=np.float32)
        output_consecutive = (
            (trajectory[1:, 0] > 0.0) & (trajectory[:-1, 0] > 0.0)
        )
        output_differences = np.diff(trajectory[:, 1:3], axis=0)
        output_dx_dy[1:][output_consecutive] = output_differences[
            output_consecutive
        ]
        trajectory[:, 5:7] = output_dx_dy

    requested = np.asarray(
        (requested_start_xy, requested_end_xy), dtype=np.float64
    )
    detector_output = (
        trajectory[[0, -1], 1:3].astype(np.float64)
        * np.asarray((width_px, height_px), dtype=np.float64)[None, :]
    )
    detector_errors = np.linalg.norm(detector_output - requested, axis=1)
    maximum_error = float(np.max(detector_errors))
    if maximum_error > 5.0e-4:
        raise ReplayDatasetBuildError(
            "exact touch template did not preserve requested endpoints"
        )
    generated_duration = float(
        (trajectory[-1, 7] - trajectory[0, 7]) * 1000.0
    )
    if not math.isclose(
        generated_duration,
        float(target_duration_ms),
        rel_tol=0.0,
        abs_tol=1.0e-3,
    ):
        raise ReplayDatasetBuildError(
            "exact touch template did not preserve requested duration"
        )
    delta = requested[1] - requested[0]
    stationary = bool(np.linalg.norm(delta) <= 1.0e-12)
    realized_direction = (
        "stationary" if stationary else str(requested_direction)
    )
    observation = TouchObservation(
        touch=trajectory[:, :7].copy(),
        trajectory=trajectory.copy(),
        touch_observed=bool(np.any(trajectory[:, 8] > 0.5)),
        source_updates=int(len(trajectory)),
    )
    return observation, {
        "conditioning_action": target.action,
        "conditioning_orientation_id": int(target.orientation_id),
        "conditioning_direction": requested_direction,
        "requested_start_px": requested[0].tolist(),
        "requested_end_px": requested[1].tolist(),
        # This path operates directly on the detector grid; the mathematical
        # pre-normalization endpoints are the exact request by construction.
        "raw_output_start_px": requested[0].tolist(),
        "raw_output_end_px": requested[1].tolist(),
        "detector_output_start_px": detector_output[0].tolist(),
        "detector_output_end_px": detector_output[1].tolist(),
        "raw_start_error_px": 0.0,
        "raw_end_error_px": 0.0,
        "detector_start_error_px": float(detector_errors[0]),
        "detector_end_error_px": float(detector_errors[1]),
        "maximum_endpoint_error_px": maximum_error,
        "requested_raw_duration_ms": float(target_duration_ms),
        "generated_raw_duration_ms": float(target_duration_ms),
        "requested_raw_row_count": int(len(trajectory)),
        "generated_raw_row_count": int(len(trajectory)),
        "residual_scale": float(generated.residual_scale),
        "transform_mode": generated.mode,
        "identity_transform": bool(generated.identity_transform),
        "tap_stationary_branch": bool(target.action == "tap" and stationary),
        "realized_direction": realized_direction,
        "generation_mode": "exact_touch_template_generator_v1",
    }


def _keystroke_target_down_anchors(
    target: AndroidTarget,
) -> tuple[tuple[float, float], ...]:
    """Return one bound target DOWN location for every requested key."""

    if target.action != "keystroke" or not target.keycodes:
        raise ReplayDatasetBuildError("keystroke target anchor request is invalid")
    ordered_indices: list[int] = []
    for raw_index in np.asarray(target.key_index, dtype=np.int64):
        index = int(raw_index)
        if index >= 0 and index not in ordered_indices:
            ordered_indices.append(index)
    if len(ordered_indices) != len(target.keycodes):
        raise ReplayDatasetBuildError(
            "keystroke target key positions do not match its keycodes"
        )
    base_actions = np.asarray(target.android_action, dtype=np.int64) & 0xFF
    anchors: list[tuple[float, float]] = []
    for index in ordered_indices:
        positions = np.flatnonzero(
            np.asarray(target.key_index, dtype=np.int64) == index
        )
        if not len(positions):
            raise ReplayDatasetBuildError("keystroke target key has no rows")
        downs = positions[base_actions[positions] == 0]
        if len(downs) != 1:
            raise ReplayDatasetBuildError(
                "keystroke target key must have one Android DOWN row"
            )
        row = int(downs[0])
        anchors.append((float(target.x_px[row]), float(target.y_px[row])))
    return tuple(anchors)


def _keystroke_reference_template(
    *,
    source_cluster_id: str,
    genuine_bindings: Mapping[str, GenuineTouchBinding],
    raw_archive_cache: dict[Path, dict[str, np.ndarray]],
) -> KeystrokeReferenceTemplate:
    """Recover a coupled genuine key sequence and its physical DOWN anchors."""

    binding = genuine_bindings.get(str(source_cluster_id))
    if binding is None or binding.action != "keystroke" or binding.split != "train":
        raise ReplayDatasetBuildError(
            "keystroke carrier reference is not a selected train genuine event"
        )
    archive = _raw_archive_cached(binding, raw_archive_cache)
    if "flat_keycode" not in archive:
        raise ReplayDatasetBuildError(
            "raw keystroke reference has no keycode binding"
        )
    index = int(binding.raw_trajectory_event_index)
    offsets = np.asarray(archive["event_offsets"], dtype=np.int64)
    left, right = (int(value) for value in offsets[index : index + 2])
    valid = np.asarray(archive["flat_valid_mask"][left:right], dtype=np.bool_)
    positions = np.arange(left, right, dtype=np.int64)[valid]
    keys = np.asarray(archive["flat_key_index"], dtype=np.int64)[positions]
    codes = np.asarray(archive["flat_keycode"], dtype=np.int64)[positions]
    actions = np.asarray(archive["flat_action_code"], dtype=np.int64)[positions] & 0xFF
    x_px = np.asarray(archive["flat_x"], dtype=np.float64)[positions]
    y_px = np.asarray(archive["flat_y"], dtype=np.float64)[positions]
    ordered: list[int] = []
    for raw_key in keys:
        key = int(raw_key)
        if key >= 0 and key not in ordered:
            ordered.append(key)
    keycodes: list[int] = []
    anchors: list[tuple[float, float]] = []
    for key in ordered:
        local = np.flatnonzero(keys == key)
        observed_codes = np.unique(codes[local])
        down = local[actions[local] == 0]
        if len(observed_codes) != 1 or len(down) != 1:
            raise ReplayDatasetBuildError(
                "raw keystroke reference has an ambiguous key binding"
            )
        row = int(down[0])
        keycodes.append(int(observed_codes[0]))
        anchors.append((float(x_px[row]), float(y_px[row])))
    if not keycodes or len(keycodes) != _binding_raw_key_count(binding, archive):
        raise ReplayDatasetBuildError(
            "raw keystroke reference key count changed"
        )
    return KeystrokeReferenceTemplate(
        source_cluster_id=str(source_cluster_id),
        source_event_id=binding.source_event_id,
        source_user_id=binding.user_id,
        source_session_id=binding.session_id,
        orientation_id=binding.orientation_id,
        keycodes=tuple(keycodes),
        down_anchors_px=tuple(anchors),
        raw_trajectory_source=binding.raw_trajectory_source,
        raw_trajectory_source_sha256=binding.raw_trajectory_source_sha256,
        raw_trajectory_event_index=binding.raw_trajectory_event_index,
        raw_event_sha256=binding.raw_event_sha256,
    )


def _plan_fake_keystroke_target(
    *,
    event_id: str,
    output_split: str,
    orientation_id: int,
    duration_sampler: "RawWindowRatioSampler",
    reference_templates: Mapping[str, KeystrokeReferenceTemplate],
    seed: int,
) -> KeystrokeTargetPlan:
    """Replace capped fake metadata with one coupled train-real reference."""

    timing = duration_sampler.sample_keystroke_carrier(
        orientation_id=int(orientation_id),
        key_count=None,
        event_id=str(event_id),
        seed=int(seed),
        output_split=output_split,
    )
    if (
        timing.reference_source_cluster_id is None
        or timing.reference_key_count is None
        or timing.reference_raw_duration_ms is None
        or timing.reference_window_duration_ms is None
        or timing.reference_window_sample_count is None
        or timing.reference_observable_update_count is None
    ):
        raise ReplayDatasetBuildError(
            "sampled keystroke reference is not a complete carrier tuple"
        )
    template = reference_templates.get(timing.reference_source_cluster_id)
    if template is None:
        raise ReplayDatasetBuildError(
            "sampled keystroke reference has no compact physical template"
        )
    if (
        template.orientation_id != int(orientation_id)
        or len(template.keycodes) != int(timing.reference_key_count)
        or len(template.down_anchors_px) != len(template.keycodes)
    ):
        raise ReplayDatasetBuildError(
            "keystroke timing and physical reference bindings disagree"
        )
    raw_duration = int(round(float(timing.reference_raw_duration_ms)))
    detector_duration = float(timing.reference_window_duration_ms)
    target_samples = int(timing.reference_window_sample_count)
    if (
        raw_duration < 1
        or detector_duration <= 0.0
        or not 2 <= target_samples <= 512
    ):
        raise ReplayDatasetBuildError("keystroke carrier tuple is invalid")
    letter_count = sum(
        _is_hmog_ascii_letter_keycode(code) for code in template.keycodes
    )
    return KeystrokeTargetPlan(
        timing=timing,
        template=template,
        raw_touch_duration_ms=raw_duration,
        detector_duration_ms=detector_duration,
        target_samples=target_samples,
        letter_count=letter_count,
    )


def _collect_keystroke_target_plans(
    *,
    request_shards_by_split: Mapping[str, Sequence[InputShard]],
    duration_sampler: "RawWindowRatioSampler",
    reference_templates: Mapping[str, KeystrokeReferenceTemplate],
    joint_events_root: Path,
    seed: int,
    request_keystrokes_per_shard: int | None = None,
    requested_keystroke_event_ids: set[str] | None = None,
) -> dict[str, KeystrokeTargetPlan]:
    """Pre-bind every selected fake keystroke to one genuine carrier."""

    if set(request_shards_by_split) != set(SPLITS):
        raise ReplayDatasetBuildError(
            "keystroke target planning needs all three fixed splits"
        )
    if request_keystrokes_per_shard is not None and int(
        request_keystrokes_per_shard
    ) < 1:
        raise ReplayDatasetBuildError(
            "keystroke request quota per shard must be positive"
        )
    if (
        request_keystrokes_per_shard is not None
        and requested_keystroke_event_ids is not None
    ):
        raise ReplayDatasetBuildError(
            "keystroke planning cannot combine count and event-ID selection"
        )
    exact_requested_ids = (
        None
        if requested_keystroke_event_ids is None
        else {str(value) for value in requested_keystroke_event_ids}
    )
    if exact_requested_ids is not None and not exact_requested_ids:
        raise ReplayDatasetBuildError(
            "exact keystroke event-ID selection is empty"
        )

    plans: dict[str, KeystrokeTargetPlan] = {}
    planned_by_split = {split: 0 for split in SPLITS}
    trajectory_cache: dict[Path, dict[str, np.ndarray]] = {}
    for split in SPLITS:
        for source in request_shards_by_split[split]:
            shard = _load_shard(source, signals=True)
            labels = np.asarray(shard.arrays["label"], dtype=np.int64)
            actions = np.asarray(shard.arrays["action"]).astype(str)
            event_ids = np.asarray(shard.arrays["event_id"]).astype(str)
            users = np.asarray(shard.arrays["user_id"]).astype(str)
            sample_indices = np.asarray(
                shard.arrays["sample_idx"], dtype=np.int64
            )
            pair_ids = np.asarray(
                shard.arrays["cross_modal_pair_id"]
            ).astype(str)
            requested_in_shard = 0
            for raw_index in np.flatnonzero(
                (labels == 1) & (actions == "keystroke")
            ):
                if (
                    request_keystrokes_per_shard is not None
                    and requested_in_shard
                    >= int(request_keystrokes_per_shard)
                ):
                    break
                index = int(raw_index)
                event_id = str(event_ids[index])
                if (
                    exact_requested_ids is not None
                    and event_id not in exact_requested_ids
                ):
                    continue
                _, old_trajectory = shard.event_signal(index)
                target = load_android_target(
                    event_id=event_id,
                    action="keystroke",
                    target_duration_ms=_detector_window_duration_ms(
                        old_trajectory
                    ),
                    joint_events_root=joint_events_root,
                    trajectory_cache=trajectory_cache,
                    expected_user_id=str(users[index]),
                    expected_split=split,
                    expected_sample_idx=int(sample_indices[index]),
                    expected_cross_modal_pair_id=str(pair_ids[index]),
                )
                if str(users[index]) != source.user_id:
                    raise ReplayDatasetBuildError(
                        "keystroke source changed shard user"
                    )
                if event_id in plans:
                    raise ReplayDatasetBuildError(
                        "duplicate keystroke target event ID"
                    )
                plans[event_id] = _plan_fake_keystroke_target(
                    event_id=event_id,
                    output_split=split,
                    orientation_id=target.orientation_id,
                    duration_sampler=duration_sampler,
                    reference_templates=reference_templates,
                    seed=seed,
                )
                requested_in_shard += 1
                planned_by_split[split] += 1
            trajectory_cache.clear()

    if not plans or any(not planned_by_split[split] for split in SPLITS):
        raise ReplayDatasetBuildError(
            "keystroke target-plan coverage is incomplete"
        )
    if exact_requested_ids is not None and set(plans) != exact_requested_ids:
        raise ReplayDatasetBuildError(
            "exact keystroke event-ID coverage is incomplete"
        )
    return plans


def _single_pointer_target_control_points_px(
    target: AndroidTarget,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Extract one exact DOWN anchor and terminal UP coordinate by lifecycle."""

    if target.action not in SINGLE_POINTER_TOUCH_ACTIONS:
        raise ReplayDatasetBuildError(
            "single-pointer control points require tap/scroll/swipe rows"
        )
    actions = np.asarray(target.android_action, dtype=np.int64) & ACTION_MASK
    pointers = np.asarray(target.pointer_id, dtype=np.int64)
    if (
        actions.shape != np.asarray(target.x_px).shape
        or pointers.shape != actions.shape
        or not len(actions)
    ):
        raise ReplayDatasetBuildError("single-pointer target rows are malformed")
    downs = np.flatnonzero(actions == ACTION_DOWN)
    ups = np.flatnonzero(actions == ACTION_UP)
    if len(downs) != 1 or len(ups) != 1 or int(ups[0]) <= int(downs[0]):
        raise ReplayDatasetBuildError(
            "single-pointer target needs one ordered DOWN/UP lifecycle"
        )
    down = int(downs[0])
    up = int(ups[0])
    if pointers[down] != pointers[up] or np.any(
        pointers[down : up + 1] != pointers[down]
    ):
        raise ReplayDatasetBuildError(
            "single-pointer target changes pointer identity"
        )
    start = (float(target.x_px[down]), float(target.y_px[down]))
    endpoint = (float(target.x_px[up]), float(target.y_px[up]))
    if not np.isfinite(np.asarray((start, endpoint), dtype=np.float64)).all():
        raise ReplayDatasetBuildError(
            "single-pointer target control point is non-finite"
        )
    return start, endpoint


def _action_target_anchor_px(target: AndroidTarget) -> tuple[float, float]:
    """Return the bound target's physical start anchor for rigid replay."""

    if target.action in {"scroll", "swipe"}:
        return _single_pointer_target_control_points_px(target)[0]
    if target.action != "pinch":
        raise ReplayDatasetBuildError(
            "physical action anchor is defined only for scroll/swipe/pinch"
        )
    frames = np.asarray(target.frame_index, dtype=np.int64)
    pointers = np.asarray(target.pointer_id, dtype=np.int64)
    actions = np.asarray(target.android_action, dtype=np.int64) & ACTION_MASK
    x_px = np.asarray(target.x_px, dtype=np.float64)
    y_px = np.asarray(target.y_px, dtype=np.float64)
    if (
        not len(frames)
        or any(value.shape != frames.shape for value in (pointers, actions, x_px, y_px))
        or np.any(np.diff(frames) < 0)
    ):
        raise ReplayDatasetBuildError("pinch target rows are malformed")
    changes = np.flatnonzero(frames[1:] != frames[:-1]) + 1
    bounds = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            changes,
            np.asarray([len(frames)], dtype=np.int64),
        )
    )
    for left, right in zip(bounds[:-1], bounds[1:]):
        positions = np.arange(int(left), int(right), dtype=np.int64)
        selected: list[int] = []
        for pointer in np.unique(pointers[positions]):
            candidates = positions[pointers[positions] == pointer]
            live = candidates[~np.isin(actions[candidates], (ACTION_UP, ACTION_CANCEL))]
            if len(live):
                selected.append(int(live[-1]))
        if len(selected) < 2:
            continue
        chosen = np.asarray(
            sorted(selected, key=lambda row: int(pointers[row]))[:2],
            dtype=np.int64,
        )
        anchor_array = np.mean(
            np.column_stack((x_px[chosen], y_px[chosen])), axis=0
        )
        if not np.isfinite(anchor_array).all():
            raise ReplayDatasetBuildError("pinch target anchor is non-finite")
        return (float(anchor_array[0]), float(anchor_array[1]))
    raise ReplayDatasetBuildError(
        "pinch target lacks a simultaneous two-pointer start anchor"
    )


def _single_pointer_target_endpoint_px(target: AndroidTarget) -> tuple[float, float]:
    """Return the requested final XY for scroll/swipe endpoint binding."""

    if target.action not in {"scroll", "swipe"}:
        raise ReplayDatasetBuildError("endpoint binding requires scroll/swipe rows")
    return _single_pointer_target_control_points_px(target)[1]


def _pinch_target_endpoint_geometry(
    target: AndroidTarget,
) -> PinchEndpointGeometry:
    """Return exact ordered live-pointer endpoint geometry for a pinch target."""

    if target.action != "pinch":
        raise ReplayDatasetBuildError(
            "pinch endpoint binding requires a pinch Android target"
        )
    try:
        return extract_live_two_pointer_endpoints(
            t_ms=target.t_ms,
            frame_index=target.frame_index,
            pointer_id=target.pointer_id,
            android_action=target.android_action,
            x_px=target.x_px,
            y_px=target.y_px,
        )
    except PinchEndpointControlError as error:
        raise ReplayDatasetBuildError(
            f"pinch target endpoint geometry is invalid: {error}"
        ) from error


def _realistic_control_endpoint_px(
    *,
    target: AndroidTarget,
    allocator: ReplayAllocator,
    event_id: str,
    seed: int,
) -> tuple[tuple[float, float], str, float]:
    """Sample an exact endpoint request from train-real geometric support."""

    start = np.rint(
        np.asarray(_action_target_anchor_px(target), dtype=np.float64)
    )
    used = set(allocator.used_primitive_ids)
    identifiers = list(allocator.pools.primitive_ids(allocator.output_split))
    digest = hashlib.sha256(
        f"gesture-control-distance|{int(seed)}|{event_id}".encode("utf-8")
    ).digest()
    if identifiers:
        offset = int.from_bytes(digest[:8], "big") % len(identifiers)
        identifiers = identifiers[offset:] + identifiers[:offset]
    for primitive_id in identifiers:
        if primitive_id in used:
            continue
        descriptor = allocator.pools.bank.descriptor(primitive_id)
        if descriptor.orientation_id != target.orientation_id:
            continue
        fitted = _fitted_isometries(
            descriptor,
            target_direction=str(target.direction),
            target_anchor_px=(float(start[0]), float(start[1])),
        )
        for isometry in fitted:
            if isometry.anchor_error_px > 1.0e-6:
                continue
            matrix = np.asarray(isometry.matrix_xy, dtype=np.float64)
            translation = np.asarray(isometry.translation_px, dtype=np.float64)
            endpoint = (
                matrix @ np.asarray(descriptor.end_xy_px, dtype=np.float64)
                + translation
            )
            if not np.allclose(endpoint, np.rint(endpoint), atol=1.0e-6):
                continue
            endpoint = np.rint(endpoint)
            distance = float(np.linalg.norm(endpoint - start))
            if distance > 1.0e-6:
                return (
                    (float(endpoint[0]), float(endpoint[1])),
                    primitive_id,
                    distance,
                )
    raise ReplayDatasetBuildError(
        "no train-real distance support fits the requested start/direction"
    )


def _output_event_provenance(
    *,
    shard: LoadedShard,
    index: int,
    input_imu: np.ndarray,
    output_imu: np.ndarray,
    old_trajectory: np.ndarray,
    observation: TouchObservation,
    method: str,
    donor: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": PROVENANCE_SCHEMA,
        "split": shard.source.split,
        "user_id": str(shard.arrays["user_id"][index]),
        "event_id": str(shard.arrays["event_id"][index]),
        "source_cluster_id": str(shard.arrays["source_cluster_id"][index]),
        "action": str(shard.arrays["action"][index]),
        "label": int(shard.arrays["label"][index]),
        "input_shard": str(shard.source.path),
        "input_shard_sha256": shard.source.sha256,
        "input_event_index": int(index),
        "input_imu_sha256": _array_sha256(input_imu),
        "output_imu_sha256": _array_sha256(output_imu),
        "input_trajectory_sha256": _array_sha256(old_trajectory),
        "output_trajectory_sha256": _array_sha256(observation.trajectory),
        "input_samples": int(len(input_imu)),
        "output_samples": int(len(output_imu)),
        "target_samples": int(len(observation.trajectory)),
        "target_duration_ms": float(observation.trajectory[-1, 7] * 1000.0),
        "input_cross_modal_pair_id": str(
            shard.arrays["cross_modal_pair_id"][index]
        ),
        "output_cross_modal_pair_id": _pair_id(
            str(shard.arrays["event_id"][index]), output_imu, observation.trajectory
        ),
        "observer": "common_android_zoh_v1",
        "linear_touch_interpolation_used": False,
        "coordinate_clipping_used": False,
        "touch_observed": bool(observation.touch_observed),
        "source_updates": int(observation.source_updates),
        "rebuild_method": method,
        "donor": dict(donor),
    }


def _raw_archive_cached(
    binding: GenuineTouchBinding,
    cache: dict[Path, dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    source = binding.raw_trajectory_source.resolve()
    if source not in cache:
        if sha256_file(source) != binding.raw_trajectory_source_sha256:
            raise ReplayDatasetBuildError(
                f"raw trajectory source hash changed: {source}"
            )
        cache[source] = load_raw_trajectory_archive(source)
    return cache[source]


def _binding_raw_duration_and_request(
    binding: GenuineTouchBinding,
    archive: Mapping[str, np.ndarray],
) -> tuple[float, str | None, str | None]:
    index = int(binding.raw_trajectory_event_index)
    event_ids = np.asarray(archive["event_id"])
    offsets = np.asarray(archive["event_offsets"], dtype=np.int64)
    if (
        not 0 <= index < len(event_ids)
        or offsets.shape != (len(event_ids) + 1,)
        or str(np.asarray(event_ids[index]).item())
        != (binding.raw_trajectory_event_id or binding.source_event_id)
    ):
        raise ReplayDatasetBuildError("raw duration binding identity changed")
    raw_duration = float(np.asarray(archive["touch_duration_ms"])[index])
    if not np.isfinite(raw_duration) or raw_duration <= 0.0:
        raise ReplayDatasetBuildError("raw touch duration is invalid")
    if binding.action not in {"scroll", "swipe", "pinch"}:
        return raw_duration, None, None
    left, right = (int(value) for value in offsets[index : index + 2])
    valid = np.asarray(archive["flat_valid_mask"][left:right], dtype=np.bool_)
    positions = np.arange(left, right, dtype=np.int64)[valid]
    if len(positions) < 2:
        raise ReplayDatasetBuildError("raw gesture has too few valid rows")
    try:
        request = classify_replay_request(
            action=binding.action,
            orientation_id=binding.orientation_id,
            target_duration_ms=raw_duration,
            x_px=archive["flat_x"][positions],
            y_px=archive["flat_y"][positions],
            pointer_id=(
                archive["flat_pointer_id"][positions]
                if binding.action == "pinch"
                else None
            ),
            android_action=(
                archive["flat_action_code"][positions]
                if binding.action == "pinch"
                else None
            ),
            frame_index=(
                archive["flat_frame_index"][positions]
                if binding.action == "pinch"
                else None
            ),
        )
    except ActionReplayError:
        # Android action labels can be fixed by an excursion waypoint even
        # when the final endpoint returns to DOWN.  Such an event remains a
        # valid duration sample for action+orientation, but not a directional
        # conditioning sample.
        return raw_duration, None, None
    return raw_duration, request.direction, request.pinch_scale_direction


def _binding_raw_key_count(
    binding: GenuineTouchBinding,
    archive: Mapping[str, np.ndarray],
) -> int | None:
    if binding.action != "keystroke":
        return None
    index = int(binding.raw_trajectory_event_index)
    offsets = np.asarray(archive["event_offsets"], dtype=np.int64)
    left, right = (int(value) for value in offsets[index : index + 2])
    valid = np.asarray(archive["flat_valid_mask"][left:right], dtype=np.bool_)
    keys = np.asarray(
        archive["flat_key_index"][left:right], dtype=np.int64
    )[valid]
    count = len(np.unique(keys[keys >= 0]))
    if count < 1:
        raise ReplayDatasetBuildError("raw keystroke timing reference has no keys")
    return int(count)


class RawWindowRatioSampler:
    """Empirical train-genuine duration and observable update conditioning."""

    def __init__(
        self,
        *,
        exact_groups: Mapping[
            tuple[str, int, str, str], Sequence[float | RawTimingReference]
        ],
        orientation_groups: Mapping[
            tuple[str, int], Sequence[float | RawTimingReference]
        ],
        raw_duration_by_cluster: Mapping[str, float],
        raw_archive_cache: Mapping[Path, dict[str, np.ndarray]],
        minimum_exact_group: int = 8,
    ) -> None:
        def normalize(
            values: Sequence[float | RawTimingReference],
        ) -> tuple[RawTimingReference, ...]:
            references = [
                value
                if isinstance(value, RawTimingReference)
                else RawTimingReference(float(value), None, None)
                for value in values
            ]
            for reference in references:
                if (
                    not np.isfinite(reference.raw_to_window_ratio)
                    or reference.raw_to_window_ratio <= 0.0
                    or (
                        reference.observable_update_rate_hz is not None
                        and (
                            not np.isfinite(reference.observable_update_rate_hz)
                            or reference.observable_update_rate_hz <= 0.0
                        )
                    )
                    or (
                        reference.raw_duration_ms is not None
                        and (
                            not np.isfinite(reference.raw_duration_ms)
                            or reference.raw_duration_ms <= 0.0
                        )
                    )
                    or (
                        reference.window_duration_ms is not None
                        and (
                            not np.isfinite(reference.window_duration_ms)
                            or reference.window_duration_ms <= 0.0
                        )
                    )
                    or (
                        reference.window_sample_count is not None
                        and reference.window_sample_count < 2
                    )
                    or (
                        reference.observable_update_count is not None
                        and reference.observable_update_count < 1
                    )
                ):
                    raise ReplayDatasetBuildError(
                        "invalid train-genuine timing reference"
                    )
            return tuple(
                sorted(
                    references,
                    key=lambda value: (
                        value.raw_to_window_ratio,
                        value.observable_update_rate_hz or -1.0,
                        value.source_cluster_id or "",
                        value.key_count or -1,
                        value.window_sample_count or -1,
                        value.window_duration_ms or -1.0,
                    ),
                )
            )

        self._exact_groups = {
            key: normalize(values) for key, values in exact_groups.items()
        }
        self._orientation_groups = {
            key: normalize(values) for key, values in orientation_groups.items()
        }
        keystroke_groups: dict[
            tuple[int, int], list[RawTimingReference]
        ] = {}
        for (action, orientation), references in self._orientation_groups.items():
            if action != "keystroke":
                continue
            for reference in references:
                if reference.key_count is not None:
                    keystroke_groups.setdefault(
                        (orientation, reference.key_count), []
                    ).append(reference)
        self._keystroke_key_groups = {
            key: normalize(values) for key, values in keystroke_groups.items()
        }
        self.raw_duration_by_cluster = dict(raw_duration_by_cluster)
        self.raw_archive_cache = dict(raw_archive_cache)
        self.minimum_exact_group = int(minimum_exact_group)
        if self.minimum_exact_group < 1:
            raise ReplayDatasetBuildError("minimum duration group must be positive")

    @classmethod
    def from_selected_train_genuine(
        cls,
        *,
        selected_clusters: Iterable[str],
        window_duration_by_cluster: Mapping[str, float],
        window_sample_count_by_cluster: Mapping[str, int] | None = None,
        genuine_bindings: Mapping[str, GenuineTouchBinding],
        minimum_exact_group: int = 8,
    ) -> "RawWindowRatioSampler":
        exact: dict[
            tuple[str, int, str, str], list[RawTimingReference]
        ] = {}
        orientation: dict[tuple[str, int], list[RawTimingReference]] = {}
        skipped_off_screen_references = 0
        raw_duration_by_cluster: dict[str, float] = {}
        by_source: dict[Path, list[tuple[str, GenuineTouchBinding]]] = {}
        for cluster_id in sorted(str(value) for value in selected_clusters):
            binding = genuine_bindings[cluster_id]
            if binding.split == "train":
                by_source.setdefault(
                    binding.raw_trajectory_source.resolve(), []
                ).append((cluster_id, binding))
        # Load one immutable archive at a time.  Retaining all train-user raw
        # archives can consume many gigabytes and is unnecessary for these
        # compact timing references.
        for source, members in sorted(by_source.items(), key=lambda item: str(item[0])):
            declared_hashes = {
                binding.raw_trajectory_source_sha256
                for _, binding in members
            }
            if len(declared_hashes) != 1 or sha256_file(source) not in declared_hashes:
                raise ReplayDatasetBuildError(
                    f"raw trajectory source hash changed: {source}"
                )
            archive = load_raw_trajectory_archive(source)
            for cluster_id, binding in members:
                raw_duration, direction, pinch_scale = (
                    _binding_raw_duration_and_request(binding, archive)
                )
                detector_window = float(
                    window_duration_by_cluster.get(cluster_id, 0.0)
                )
                detector_samples = int(
                    (
                        window_sample_count_by_cluster.get(cluster_id, 0)
                        if window_sample_count_by_cluster is not None
                        else max(2, int(round(detector_window / 10.0)) + 1)
                    )
                )
                if detector_window <= 0.0:
                    raise ReplayDatasetBuildError(
                        "genuine detector window is invalid"
                    )
                ratio = raw_duration / detector_window
                if not np.isfinite(ratio) or ratio <= 0.0:
                    raise ReplayDatasetBuildError(
                        "raw/window duration ratio is invalid"
                    )
                try:
                    raw_observation = observe_genuine_binding(
                        binding,
                        target_samples=max(2, int(round(raw_duration / 10.0)) + 1),
                        target_duration_ms=raw_duration,
                        archive=archive,
                    )
                except TouchObservationError:
                    # A handful of real HMOG recordings report a coordinate just
                    # off the panel; coordinates are never clipped here, so such
                    # an event contributes no timing reference rather than
                    # stopping the build over a property of the recording.
                    skipped_off_screen_references += 1
                    continue
                update_rate = (
                    float(raw_observation.source_updates) * 1000.0
                    / detector_window
                )
                if not np.isfinite(update_rate) or update_rate <= 0.0:
                    raise ReplayDatasetBuildError(
                        "raw observable update rate is invalid"
                    )
                reference = RawTimingReference(
                    raw_to_window_ratio=ratio,
                    observable_update_rate_hz=update_rate,
                    source_cluster_id=cluster_id,
                    key_count=_binding_raw_key_count(binding, archive),
                    raw_duration_ms=raw_duration,
                    window_duration_ms=detector_window,
                    window_sample_count=detector_samples,
                    observable_update_count=int(raw_observation.source_updates),
                )
                raw_duration_by_cluster[cluster_id] = raw_duration
                orientation.setdefault(
                    (binding.action, binding.orientation_id), []
                ).append(reference)
                if direction is not None:
                    exact.setdefault(
                        (
                            binding.action,
                            binding.orientation_id,
                            direction,
                            pinch_scale or "-",
                        ),
                        [],
                    ).append(reference)
        if not raw_duration_by_cluster:
            raise ReplayDatasetBuildError("no selected train genuine duration ratios")
        return cls(
            exact_groups=exact,
            orientation_groups=orientation,
            raw_duration_by_cluster=raw_duration_by_cluster,
            raw_archive_cache={},
            minimum_exact_group=minimum_exact_group,
        )

    def sample(
        self,
        *,
        action: str,
        orientation_id: int,
        event_id: str,
        seed: int,
        direction: str | None = None,
        pinch_scale_direction: str | None = None,
        target_duration_ms: float | None = None,
        key_count: int | None = None,
    ) -> DurationRatioSample:
        exact_key = (
            str(action),
            int(orientation_id),
            direction or "-",
            pinch_scale_direction or "-",
        )
        exact_values = self._exact_groups.get(exact_key, ())
        if action == "keystroke":
            if key_count is None or int(key_count) < 1:
                raise ReplayDatasetBuildError(
                    "keystroke timing sampling needs a positive target key count"
                )
            if target_duration_ms is None or target_duration_ms <= 0.0:
                raise ReplayDatasetBuildError(
                    "keystroke timing sampling needs the detector duration"
                )
            count = int(key_count)
            minimum = (
                count * ROBUST_KEYSTROKE_BOUNDS.hold_min_ms
                + max(0, count - 1)
                * ROBUST_KEYSTROKE_BOUNDS.flight_min_ms
            )
            maximum = (
                count * ROBUST_KEYSTROKE_BOUNDS.hold_max_ms
                + max(0, count - 1)
                * ROBUST_KEYSTROKE_BOUNDS.flight_max_ms
            )

            def feasible(
                references: Sequence[RawTimingReference],
            ) -> tuple[RawTimingReference, ...]:
                return tuple(
                    reference
                    for reference in references
                    if minimum
                    <= int(
                        round(
                            float(target_duration_ms)
                            * reference.raw_to_window_ratio
                        )
                    )
                    <= maximum
                )

            values = feasible(
                self._keystroke_key_groups.get(
                    (int(orientation_id), count), ()
                )
            )
            if values:
                conditioning = "action_orientation_key_count_feasible"
            else:
                values = feasible(
                    self._orientation_groups.get(
                        (str(action), int(orientation_id)), ()
                    )
                )
                conditioning = "action_orientation_feasible_for_key_count"
        elif direction is not None and len(exact_values) >= self.minimum_exact_group:
            values = exact_values
            conditioning = "action_orientation_direction_pinch_scale"
        else:
            values = self._orientation_groups.get(
                (str(action), int(orientation_id)), ()
            )
            conditioning = "action_orientation"
        if not values:
            raise ReplayDatasetBuildError(
                f"no train-genuine raw/window ratios for {action}/{orientation_id}"
            )
        digest = hashlib.sha256(
            f"duration-ratio|{int(seed)}|{event_id}".encode("utf-8")
        ).digest()
        index = int.from_bytes(digest[:8], "big") % len(values)
        return DurationRatioSample(
            raw_to_window_ratio=float(values[index].raw_to_window_ratio),
            target_update_rate_hz=values[index].observable_update_rate_hz,
            reference_source_cluster_id=values[index].source_cluster_id,
            reference_key_count=values[index].key_count,
            conditioning=conditioning,
            conditioning_count=len(values),
            reference_raw_duration_ms=values[index].raw_duration_ms,
            reference_window_duration_ms=values[index].window_duration_ms,
            reference_window_sample_count=values[index].window_sample_count,
            reference_observable_update_count=(
                values[index].observable_update_count
            ),
        )

    def sample_keystroke_carrier(
        self,
        *,
        orientation_id: int,
        key_count: int | None,
        event_id: str,
        seed: int,
        output_split: str | None = None,
    ) -> DurationRatioSample:
        """Sample one complete train-genuine timing/carrier tuple."""

        count = None if key_count is None else int(key_count)
        if output_split is not None and output_split not in SPLITS:
            raise ReplayDatasetBuildError(
                f"unknown keystroke carrier output split {output_split}"
            )
        if count is not None and count < 1:
            raise ReplayDatasetBuildError(
                "keystroke carrier sampling needs a positive key count"
            )

        def complete_and_feasible(
            references: Sequence[RawTimingReference],
        ) -> tuple[RawTimingReference, ...]:
            selected: list[RawTimingReference] = []
            for reference in references:
                reference_count = reference.key_count
                if reference_count is None or reference_count < 1:
                    continue
                minimum = (
                    reference_count * ROBUST_KEYSTROKE_BOUNDS.hold_min_ms
                    + max(0, reference_count - 1)
                    * ROBUST_KEYSTROKE_BOUNDS.flight_min_ms
                )
                maximum = (
                    reference_count * ROBUST_KEYSTROKE_BOUNDS.hold_max_ms
                    + max(0, reference_count - 1)
                    * ROBUST_KEYSTROKE_BOUNDS.flight_max_ms
                )
                if (
                    (
                        output_split is None
                        or (
                            reference.source_cluster_id is not None
                            and _donor_output_split(
                                "keystroke-carrier-reference|"
                                + str(reference.source_cluster_id),
                                seed=seed,
                            )
                            == output_split
                        )
                    )
                    and reference.raw_duration_ms is not None
                    and minimum
                    <= int(round(reference.raw_duration_ms))
                    <= maximum
                    and reference.window_duration_ms is not None
                    # The five-shot composer fills the DETECTOR window, not the
                    # raw touch span, so the window has to clear the same key
                    # schedule bounds.  A handful of genuine references type
                    # slowly enough that their window exceeds what the robust
                    # flight bound can span; offering one would hand the
                    # composer a request it cannot satisfy.
                    and minimum
                    <= int(round(reference.window_duration_ms))
                    <= maximum
                    and reference.window_sample_count is not None
                    and reference.observable_update_count is not None
                ):
                    selected.append(reference)
            return tuple(selected)

        if count is None:
            values = complete_and_feasible(
                self._orientation_groups.get(
                    ("keystroke", int(orientation_id)), ()
                )
            )
            conditioning = "action_orientation_complete_carrier"
        else:
            values = complete_and_feasible(
                self._keystroke_key_groups.get(
                    (int(orientation_id), count), ()
                )
            )
            if values:
                conditioning = "action_orientation_key_count_complete_carrier"
            else:
                values = complete_and_feasible(
                    self._orientation_groups.get(
                        ("keystroke", int(orientation_id)), ()
                    )
                )
                conditioning = "action_orientation_feasible_complete_carrier"
        if not values:
            raise ReplayDatasetBuildError(
                "no complete train-genuine keystroke carrier reference for "
                f"orientation={int(orientation_id)}, key_count={count}"
            )
        digest = hashlib.sha256(
            f"keystroke-carrier|{int(seed)}|{event_id}".encode("utf-8")
        ).digest()
        reference = values[int.from_bytes(digest[:8], "big") % len(values)]
        return DurationRatioSample(
            raw_to_window_ratio=float(reference.raw_to_window_ratio),
            target_update_rate_hz=reference.observable_update_rate_hz,
            reference_source_cluster_id=reference.source_cluster_id,
            reference_key_count=reference.key_count,
            conditioning=conditioning,
            conditioning_count=len(values),
            reference_raw_duration_ms=reference.raw_duration_ms,
            reference_window_duration_ms=reference.window_duration_ms,
            reference_window_sample_count=reference.window_sample_count,
            reference_observable_update_count=(
                reference.observable_update_count
            ),
        )

    @property
    def keystroke_reference_cluster_ids(self) -> tuple[str, ...]:
        values = {
            str(reference.source_cluster_id)
            for (action, _), references in self._orientation_groups.items()
            if action == "keystroke"
            for reference in references
            if reference.source_cluster_id is not None
        }
        return tuple(sorted(values))

    def summary(self) -> dict[str, Any]:
        groups: dict[str, Any] = {}
        for (action, orientation), values in sorted(self._orientation_groups.items()):
            array = np.asarray(
                [value.raw_to_window_ratio for value in values], dtype=np.float64
            )
            rates = np.asarray(
                [
                    value.observable_update_rate_hz
                    for value in values
                    if value.observable_update_rate_hz is not None
                ],
                dtype=np.float64,
            )
            raw_durations = np.asarray(
                [
                    value.raw_duration_ms
                    for value in values
                    if value.raw_duration_ms is not None
                ],
                dtype=np.float64,
            )
            window_durations = np.asarray(
                [
                    value.window_duration_ms
                    for value in values
                    if value.window_duration_ms is not None
                ],
                dtype=np.float64,
            )
            window_samples = np.asarray(
                [
                    value.window_sample_count
                    for value in values
                    if value.window_sample_count is not None
                ],
                dtype=np.int64,
            )
            groups[f"{action}/{orientation}"] = {
                "count": len(values),
                "q05": float(np.quantile(array, 0.05)),
                "median": float(np.quantile(array, 0.5)),
                "q95": float(np.quantile(array, 0.95)),
                "observable_update_rate_hz": (
                    {
                        "count": len(rates),
                        "q05": float(np.quantile(rates, 0.05)),
                        "median": float(np.quantile(rates, 0.5)),
                        "q95": float(np.quantile(rates, 0.95)),
                    }
                    if len(rates)
                    else None
                ),
                "raw_duration_ms": (
                    {
                        "count": len(raw_durations),
                        "q05": float(np.quantile(raw_durations, 0.05)),
                        "median": float(np.quantile(raw_durations, 0.5)),
                        "q95": float(np.quantile(raw_durations, 0.95)),
                    }
                    if len(raw_durations)
                    else None
                ),
                "window_duration_ms": (
                    {
                        "count": len(window_durations),
                        "q05": float(np.quantile(window_durations, 0.05)),
                        "median": float(np.quantile(window_durations, 0.5)),
                        "q95": float(np.quantile(window_durations, 0.95)),
                    }
                    if len(window_durations)
                    else None
                ),
                "window_sample_count": (
                    {
                        "count": len(window_samples),
                        "minimum": int(np.min(window_samples)),
                        "median": float(np.quantile(window_samples, 0.5)),
                        "maximum": int(np.max(window_samples)),
                        "count_512": int(np.sum(window_samples == 512)),
                    }
                    if len(window_samples)
                    else None
                ),
            }
        return {
            "source": "selected_train_genuine_only",
            "conditioning_minimum_exact_group": self.minimum_exact_group,
            "selected_train_events": len(self.raw_duration_by_cluster),
            "action_orientation_groups": groups,
        }


class ReplayContext:
    def __init__(
        self,
        *,
        action_banks: Mapping[str, ActionReplayBank],
        action_allocators: Mapping[tuple[str, str], ReplayAllocator],
        tap_allocators: Mapping[str, TapReplayAllocator],
        keystroke_target_plans: Mapping[str, KeystrokeTargetPlan],
        genuine_bindings: Mapping[str, GenuineTouchBinding],
        duration_sampler: RawWindowRatioSampler,
        joint_events_root: Path,
        seed: int,
        tap_strategy: str = "train_raw_tap_replay",
        conditional_touch_generator: ConditionalTouchGenerator | None = None,
        conditional_touch_model_path: Path | None = None,
        conditional_touch_request_generator: (
            ConditionalTouchRequestGenerator | None
        ) = None,
        conditional_touch_request_model_path: Path | None = None,
        smoke_touch_requests: Mapping[str, SmokeTouchRequest] | None = None,
        smoke_touch_templates: Mapping[str, np.ndarray] | None = None,
        fiveshot_material: FiveShotMaterialPool | None = None,
    ) -> None:
        self.action_banks = dict(action_banks)
        self.action_allocators = dict(action_allocators)
        self.tap_allocators = dict(tap_allocators)
        self.keystroke_target_plans = dict(keystroke_target_plans)
        self.genuine_bindings = dict(genuine_bindings)
        self.duration_sampler = duration_sampler
        self.joint_events_root = joint_events_root
        self.seed = int(seed)
        self.conditional_touch_generator = conditional_touch_generator
        self.conditional_touch_model_path = (
            None
            if conditional_touch_model_path is None
            else Path(conditional_touch_model_path).resolve()
        )
        self.conditional_touch_request_generator = (
            conditional_touch_request_generator
        )
        self.conditional_touch_request_model_path = (
            None
            if conditional_touch_request_model_path is None
            else Path(conditional_touch_request_model_path).resolve()
        )
        self.fiveshot_material = fiveshot_material
        # Filled per shard by plan_fiveshot_assignment(); maps one output event
        # to the frozen donor it must transform.
        self.material_assignment: dict[str, MaterialEvent] = {}
        # Filled alongside it for the actions whose duration is read off the
        # victim's material instead of inherited from the carrier.
        self.fiveshot_timing: dict[str, _GestureTiming] = {}
        self._fiveshot_duration_laws: dict[
            tuple[str, str], GestureDurationLaw
        ] = {}
        self._keystroke_rhythms: dict[str, KeystrokeRhythm] = {}
        self._keystroke_pulse_models: dict[str, KeystrokeImuPulseModel] = {}
        self.smoke_touch_requests = dict(smoke_touch_requests or {})
        self.smoke_touch_templates = {
            str(event_id): np.asarray(value, dtype=np.float32).copy()
            for event_id, value in (smoke_touch_templates or {}).items()
        }
        if tap_strategy not in {
            "train_raw_tap_replay",
            "bound_fake_tap_android_zoh",
            "conditional_touch_generator",
        }:
            raise ReplayDatasetBuildError(f"unknown tap strategy {tap_strategy}")
        if (tap_strategy == "conditional_touch_generator") != (
            conditional_touch_generator is not None
        ):
            raise ReplayDatasetBuildError(
                "conditional touch strategy/model configuration is inconsistent"
            )
        if conditional_touch_generator is not None and (
            self.conditional_touch_model_path is None
            or not self.conditional_touch_model_path.is_file()
        ):
            raise ReplayDatasetBuildError(
                "conditional touch model artifact is missing"
            )
        if (conditional_touch_request_generator is None) != (
            self.conditional_touch_request_model_path is None
        ):
            raise ReplayDatasetBuildError(
                "conditional touch request model configuration is inconsistent"
            )
        if (
            conditional_touch_request_generator is not None
            and (
                conditional_touch_generator is None
                or not self.conditional_touch_request_model_path.is_file()
            )
        ):
            raise ReplayDatasetBuildError(
                "conditional touch request model artifact is missing"
            )
        self.conditional_touch_model_file_sha256: str | None = None
        self.conditional_touch_model_artifact_sha256: str | None = None
        self.conditional_touch_training_summary: dict[str, Any] | None = None
        self.conditional_touch_training_summary_sha256: str | None = None
        self.conditional_touch_request_model_file_sha256: str | None = None
        self.conditional_touch_request_model_artifact_sha256: str | None = None
        self.conditional_touch_request_training_summary: dict[str, Any] | None = None
        if conditional_touch_generator is not None:
            self.conditional_touch_model_file_sha256 = sha256_file(
                self.conditional_touch_model_path
            )
            self.conditional_touch_model_artifact_sha256 = (
                conditional_touch_generator.artifact_sha256
            )
            if (
                not isinstance(
                    self.conditional_touch_model_artifact_sha256, str
                )
                or len(self.conditional_touch_model_artifact_sha256) != 64
            ):
                raise ReplayDatasetBuildError(
                    "conditional touch model canonical artifact digest is invalid"
                )
            self.conditional_touch_training_summary = dict(
                conditional_touch_generator.training_summary
            )
            try:
                encoded_summary = json.dumps(
                    self.conditional_touch_training_summary,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            except (TypeError, ValueError) as exc:
                raise ReplayDatasetBuildError(
                    "conditional touch training summary is not JSON serializable"
                ) from exc
            self.conditional_touch_training_summary_sha256 = hashlib.sha256(
                encoded_summary
            ).hexdigest()
        if conditional_touch_request_generator is not None:
            self.conditional_touch_request_model_file_sha256 = sha256_file(
                self.conditional_touch_request_model_path
            )
            self.conditional_touch_request_model_artifact_sha256 = (
                conditional_touch_request_generator.artifact_sha256
            )
            if (
                self.conditional_touch_request_model_artifact_sha256
                != self.conditional_touch_request_model_file_sha256
            ):
                raise ReplayDatasetBuildError(
                    "conditional touch request model artifact digest is invalid"
                )
            self.conditional_touch_request_training_summary = dict(
                conditional_touch_request_generator.training_summary
            )
        self.tap_strategy = tap_strategy
        self.trajectory_cache: dict[Path, dict[str, np.ndarray]] = {}
        self.raw_archive_cache = dict(duration_sampler.raw_archive_cache)
        self._persistent_raw_archive_sources: set[Path] = set()
        self._persistent_raw_archive_sha256: dict[Path, str] = {}
        for bank in self.action_banks.values():
            source = bank.source_path.resolve()
            self.raw_archive_cache[source] = bank.common_observer_archive()
            self._persistent_raw_archive_sources.add(source)
            self._persistent_raw_archive_sha256[source] = bank.source_sha256

    def clear_transient_caches(self) -> None:
        for source in list(self.raw_archive_cache):
            if source not in self._persistent_raw_archive_sources:
                self.raw_archive_cache.pop(source)
        self.trajectory_cache.clear()

    def keystroke_rhythm(self, user_id: str) -> KeystrokeRhythm | None:
        """Return this victim's own typing rhythm, built once and held."""

        if self.fiveshot_material is None:
            return None
        cached = self._keystroke_rhythms.get(str(user_id))
        if cached is not None:
            return cached
        try:
            shots = self.fiveshot_material.shots(
                user_id=str(user_id), action="keystroke"
            )
        except FiveShotMaterialError:
            return None
        rhythm = rhythm_from_material(shots)
        self._keystroke_rhythms[str(user_id)] = rhythm
        return rhythm

    def keystroke_pulse_model(self, user_id: str) -> KeystrokeImuPulseModel | None:
        """Fit this victim's own typing IMU adapter once, then hold it."""

        if self.fiveshot_material is None:
            return None
        cached = self._keystroke_pulse_models.get(str(user_id))
        if cached is not None:
            return cached
        manifest = self.fiveshot_material.release.get("material_manifest")
        if not manifest:
            raise ReplayDatasetBuildError(
                "five-shot material release does not name its manifest"
            )
        try:
            sources = load_user_pulse_sources(str(manifest), user_id=str(user_id))
            model = fit_keystroke_imu_pulse_model(sources, user_id=str(user_id))
        except KeystrokeImuPulseError as exc:
            raise ReplayDatasetBuildError(
                f"five-shot keystroke IMU adapter could not be fitted: {exc}"
            ) from exc
        self._keystroke_pulse_models[str(user_id)] = model
        return model

    def plan_fiveshot_assignment(self, shard: LoadedShard) -> dict[str, int]:
        """Assign this shard's fake single-pointer events to frozen donors."""

        self.material_assignment = {}
        self.fiveshot_timing = {}
        if self.fiveshot_material is None:
            return {}
        actions = np.asarray(shard.arrays["action"]).astype(str)
        labels = np.asarray(shard.arrays["label"], dtype=np.int64)
        event_ids = np.asarray(shard.arrays["event_id"]).astype(str)
        users = np.asarray(shard.arrays["user_id"]).astype(str)
        planned: dict[str, int] = {}
        for action in sorted(
            EXACT_TOUCH_TEMPLATE_ACTIONS | PINCH_AREA_SIMILARITY_ACTIONS
        ):
            selected = np.flatnonzero((actions == action) & (labels == 1))
            if not len(selected):
                continue
            user_id = str(users[int(selected[0])])
            if len(set(users[selected].tolist())) != 1:
                raise ReplayDatasetBuildError("shard mixes users in one action")
            if action in FIVESHOT_TIMING_ACTIONS:
                # Planned before the donors are dealt: the timing decides how
                # many rows each gesture gets, and the donor is matched on that
                # count, so reading the carrier's count here would pair against
                # a length no gesture ends up having.
                self.fiveshot_timing.update(
                    self._plan_fiveshot_timing(
                        shard, selected, action=action, user_id=user_id
                    )
                )
            if action in PINCH_AREA_SIMILARITY_ACTIONS:
                assignment = self.fiveshot_material.plan_matched(
                    user_id=user_id,
                    action=action,
                    event_scale=self._pinch_requested_spans(shard, selected),
                    shot_scale={
                        int(shot.shot_ordinal): _pinch_material_widest_span(shot)
                        for shot in self.fiveshot_material.shots(
                            user_id=user_id, action=action
                        )
                    },
                    seed=self.seed,
                )
            else:
                assignment = self.fiveshot_material.plan_matched(
                    user_id=user_id,
                    action=action,
                    event_scale=self._carrier_sample_counts(shard, selected),
                    shot_scale={
                        int(shot.shot_ordinal): float(shot.samples)
                        for shot in self.fiveshot_material.shots(
                            user_id=user_id, action=action
                        )
                    },
                    seed=self.seed,
                )
            self.material_assignment.update(assignment)
            planned[action] = int(len(selected))
        return planned

    def _place_pinch(
        self,
        material: Any,
        target: AndroidTarget,
        *,
        target_samples: int,
        target_duration_ms: float,
        requested: PinchEndpointGeometry,
        user_id: str,
    ) -> tuple[TouchObservation, dict[str, Any], Any]:
        """Place the assigned pinch, retrying only what the request allows."""

        if self.fiveshot_material is None:
            raise ReplayDatasetBuildError("pinch placement needs frozen material")
        shots = self.fiveshot_material.shots(user_id=user_id, action="pinch")
        assigned_span = _pinch_material_widest_span(material)
        # Fall back to the least-used feasible recording rather than the closest
        # one in size: a request that cannot take its assigned shot is rare, and
        # spreading those few over the quietest shots keeps reuse even instead of
        # turning one recording into a hotspot.
        candidates = [material] + sorted(
            (shot for shot in shots if shot.event_id != material.event_id),
            key=lambda shot: (
                self.fiveshot_material.uses(
                    user_id=user_id, action="pinch", shot_ordinal=shot.shot_ordinal
                ),
                abs(_pinch_material_widest_span(shot) - assigned_span),
                int(shot.shot_ordinal),
            ),
        )
        failure: ReplayDatasetBuildError | None = None
        for candidate in candidates:
            for pointer_order in PINCH_POINTER_ORDERS:
                try:
                    observation, generation = _observe_pinch_area_similarity(
                        candidate,
                        target,
                        target_samples=target_samples,
                        target_duration_ms=target_duration_ms,
                        requested=requested,
                        pointer_order=pointer_order,
                    )
                except ReplayDatasetBuildError as exc:
                    failure = exc
                    continue
                if candidate.event_id != material.event_id:
                    self.fiveshot_material.record_substitution(
                        user_id=user_id,
                        action="pinch",
                        previous=int(material.shot_ordinal),
                        chosen=int(candidate.shot_ordinal),
                    )
                generation["material_substituted"] = bool(
                    candidate.event_id != material.event_id
                )
                generation["assigned_material_shot_ordinal"] = int(
                    material.shot_ordinal
                )
                return observation, generation, candidate
        raise ReplayDatasetBuildError(
            f"no frozen pinch of {user_id} can be placed on this area: {failure}"
        )

    def _carrier_sample_counts(
        self, shard: LoadedShard, selected: np.ndarray
    ) -> dict[str, float]:
        """Read how many detector samples each fake single-pointer event gets."""

        event_ids = np.asarray(shard.arrays["event_id"]).astype(str)
        counts: dict[str, float] = {}
        for index in selected.tolist():
            index = int(index)
            event_id = str(event_ids[index])
            timing = self.fiveshot_timing.get(event_id)
            if timing is not None:
                counts[event_id] = float(timing.samples)
                continue
            imu, _ = shard.event_signal(index)
            counts[event_id] = float(len(imu))
        return counts

    def _fiveshot_duration_law(
        self, *, user_id: str, action: str, width_px: float, height_px: float
    ) -> GestureDurationLaw:
        """Read this victim's own travel-to-duration curve, once per victim."""

        key = (str(user_id), str(action))
        law = self._fiveshot_duration_laws.get(key)
        if law is not None:
            return law
        shots = list(
            self.fiveshot_material.shots(user_id=user_id, action=action)
        )
        try:
            law = law_from_material(
                shots, width_px=float(width_px), height_px=float(height_px)
            )
        except FiveShotGestureTimingError as exc:
            raise ReplayDatasetBuildError(
                f"five-shot {action} material of {user_id} carries no timing: {exc}"
            ) from exc
        self._fiveshot_duration_laws[key] = law
        return law

    def _plan_fiveshot_timing(
        self,
        shard: LoadedShard,
        selected: np.ndarray,
        *,
        action: str,
        user_id: str,
    ) -> dict[str, "_GestureTiming"]:
        """Decide how long every fake gesture of one action takes."""

        event_ids = np.asarray(shard.arrays["event_id"]).astype(str)
        sample_indices = np.asarray(shard.arrays["sample_idx"], dtype=np.int64)
        pair_ids = np.asarray(shard.arrays["cross_modal_pair_id"]).astype(str)
        planned: dict[str, _GestureTiming] = {}
        for index in selected.tolist():
            index = int(index)
            event_id = str(event_ids[index])
            _, old_trajectory = shard.event_signal(index)
            target = load_android_target(
                event_id=event_id,
                action=action,
                target_duration_ms=_detector_window_duration_ms(old_trajectory),
                joint_events_root=self.joint_events_root,
                trajectory_cache=self.trajectory_cache,
                expected_user_id=user_id,
                expected_split=shard.source.split,
                expected_sample_idx=int(sample_indices[index]),
                expected_cross_modal_pair_id=str(pair_ids[index]),
            )
            width_px, height_px = screen_dimensions_for_orientation(
                target.orientation_id
            )
            start_px, end_px = _single_pointer_target_control_points_px(target)
            travel = float(
                math.hypot(end_px[0] - start_px[0], end_px[1] - start_px[1])
            )
            law = self._fiveshot_duration_law(
                user_id=user_id,
                action=action,
                width_px=float(width_px),
                height_px=float(height_px),
            )
            imu_source = _carrier_imu_sources(str(target.trajectory_source))[
                int(target.trajectory_archive_index)
            ]
            window, _mask = _carrier_imu_window(imu_source)
            offset = 0.0
            draw = -1
            chosen = -1
            if FIVESHOT_TIMING_SPREAD == "loo_residual":
                # Drawn from the event id so the same event lands on the same
                # departure in every worker and every rebuild, and so two events
                # of one victim at one travel do not share a duration.
                draw = int(
                    hashlib.sha256(
                        f"{self.seed}:{event_id}".encode()
                    ).hexdigest()[:16],
                    16,
                )
                offset = law.residual(draw)
                chosen = law.residual_index(draw)
            asked = law.duration_ms(travel, log_offset=offset)
            held = bool(asked < FIVESHOT_TIMING_FLOOR_MS)
            wanted = _fiveshot_timing_samples(
                max(asked, FIVESHOT_TIMING_FLOOR_MS)
            )
            samples = min(wanted, int(len(window)))
            if samples < 2:
                raise ReplayDatasetBuildError(
                    f"five-shot timing produced an unreportable {action} gesture"
                )
            planned[event_id] = _GestureTiming(
                travel_px=travel,
                # Never `(samples - 1) * period`: the observer's clock keeps the
                # rounding signature of whatever filled it, and an exact decimal
                # is a signature no genuine event carries.
                duration_ms=detector_grid_span_ms(samples),
                samples=int(samples),
                requested_samples=int(wanted),
                imu_source=str(imu_source),
                law_source_event_ids=law.source_event_ids,
                spread_policy=FIVESHOT_TIMING_SPREAD,
                log_offset=float(offset),
                residual_draw=int(draw),
                residual_index=int(chosen),
                residual_spread=float(law.residual_spread),
                floored_to_reportable=held,
            )
        return planned

    def _pinch_requested_spans(
        self, shard: LoadedShard, selected: np.ndarray
    ) -> dict[str, float]:
        """Read the area span every fake pinch is asked to operate on."""

        event_ids = np.asarray(shard.arrays["event_id"]).astype(str)
        users = np.asarray(shard.arrays["user_id"]).astype(str)
        samples = np.asarray(shard.arrays["sample_idx"], dtype=np.int64)
        pairs = np.asarray(shard.arrays["cross_modal_pair_id"]).astype(str)
        spans: dict[str, float] = {}
        for index in selected.tolist():
            index = int(index)
            event_id = str(event_ids[index])
            _, old_trajectory = shard.event_signal(index)
            target = load_android_target(
                event_id=event_id,
                action="pinch",
                target_duration_ms=_detector_window_duration_ms(old_trajectory),
                joint_events_root=self.joint_events_root,
                trajectory_cache=self.trajectory_cache,
                expected_user_id=str(users[index]),
                expected_split=shard.source.split,
                expected_sample_idx=int(samples[index]),
                expected_cross_modal_pair_id=str(pairs[index]),
            )
            spans[event_id] = _pinch_requested_widest_span(
                _pinch_target_endpoint_geometry(target)
            )
        return spans

    def rebuild(
        self,
        *,
        shard: LoadedShard,
        index: int,
    ) -> tuple[RebuiltEventSignal, dict[str, Any]]:
        imu, old_trajectory = shard.event_signal(index)
        action = str(shard.arrays["action"][index])
        label = int(shard.arrays["label"][index])
        event_id = str(shard.arrays["event_id"][index])
        input_samples = len(imu)
        output_imu = np.asarray(imu, dtype=np.float32)
        if input_samples < 2 or old_trajectory.shape != (input_samples, 9):
            raise ReplayDatasetBuildError("input event signal shapes changed")
        input_duration_ms = float(old_trajectory[-1, 7] * 1000.0)
        if label == 0:
            cluster_id = str(shard.arrays["source_cluster_id"][index])
            binding = self.genuine_bindings.get(cluster_id)
            if binding is None or binding.action != action or binding.split != shard.source.split:
                raise ReplayDatasetBuildError(
                    f"genuine event has no exact raw binding: {cluster_id}"
                )
            raw_archive = _raw_archive_cached(binding, self.raw_archive_cache)
            persistent_digest = self._persistent_raw_archive_sha256.get(
                binding.raw_trajectory_source.resolve()
            )
            if (
                persistent_digest is not None
                and persistent_digest != binding.raw_trajectory_source_sha256
            ):
                raise ReplayDatasetBuildError(
                    "persistent raw action bank hash differs from genuine binding"
                )
            raw_duration = _binding_raw_duration_and_request(
                binding, raw_archive
            )[0]
            retained_endpoint = float(old_trajectory[-1, 7] * 1000.0)
            repaired_carrier_time = retained_endpoint <= 0.0
            target_duration = _detector_window_duration_ms(
                old_trajectory,
                fallback_duration_ms=(
                    raw_duration if repaired_carrier_time else None
                ),
            )
            observation = observe_genuine_binding(
                binding,
                target_samples=input_samples,
                target_duration_ms=target_duration,
                archive=raw_archive,
            )
            donor = {
                "role": "self_raw_genuine",
                "source_event_id": binding.source_event_id,
                "source_user_id": binding.user_id,
                "raw_trajectory_source": str(binding.raw_trajectory_source),
                "raw_trajectory_source_sha256": binding.raw_trajectory_source_sha256,
                "raw_trajectory_event_index": binding.raw_trajectory_event_index,
                "raw_event_sha256": binding.raw_event_sha256,
                "raw_duration_ms": raw_duration,
                "detector_window_duration_ms": target_duration,
                "retained_carrier_time_endpoint_ms": retained_endpoint,
                "carrier_time_repaired_from_genuine_binding": (
                    repaired_carrier_time
                ),
                "carrier_time_recovery_source": (
                    "quality_bound_raw_touch_duration_ms"
                    if repaired_carrier_time
                    else None
                ),
            }
            method = "raw_genuine_recovery"
        else:
            input_target_duration = _detector_window_duration_ms(old_trajectory)
            target_duration = input_target_duration
            target_samples = input_samples
            target = load_android_target(
                event_id=event_id,
                action=action,
                target_duration_ms=input_target_duration,
                joint_events_root=self.joint_events_root,
                trajectory_cache=self.trajectory_cache,
                expected_user_id=str(shard.arrays["user_id"][index]),
                expected_split=shard.source.split,
                expected_sample_idx=int(shard.arrays["sample_idx"][index]),
                expected_cross_modal_pair_id=str(
                    shard.arrays["cross_modal_pair_id"][index]
                ),
            )
            target_binding = {
                "trajectory_source": str(target.trajectory_source),
                "trajectory_source_sha256": target.trajectory_source_sha256,
                "trajectory_archive_index": target.trajectory_archive_index,
                "orientation_id": target.orientation_id,
                "direction": target.direction,
                "pinch_scale_direction": target.pinch_scale_direction,
                "keycodes": list(target.keycodes),
                "bound_event_plan_sha256": target.bound_event_plan_sha256,
                "duration_source": "retained_trajectory_elapsed_endpoint",
                "duration_ms": input_target_duration,
            }
            gesture_timing = self.fiveshot_timing.get(event_id)
            if gesture_timing is not None:
                if action != target.action:
                    raise ReplayDatasetBuildError(
                        "planned gesture timing does not match the bound action"
                    )
                window, mask = _carrier_imu_window(gesture_timing.imu_source)
                output_imu, imu_audit = carrier_window_imu(
                    window=window, mask=mask, samples=gesture_timing.samples
                )
                target_duration = gesture_timing.duration_ms
                target_samples = gesture_timing.samples
                target_binding["duration_source"] = (
                    "fiveshot_material_travel_duration_law"
                )
                target_binding["duration_ms"] = target_duration
                target_binding["fiveshot_gesture_timing"] = {
                    "requested_travel_px": gesture_timing.travel_px,
                    "law_source_event_ids": list(
                        gesture_timing.law_source_event_ids
                    ),
                    "carrier_duration_ms": input_target_duration,
                    "carrier_sample_count": int(input_samples),
                    "requested_samples": int(gesture_timing.requested_samples),
                    "capped_to_carrier_window": bool(
                        gesture_timing.capped_to_window
                    ),
                    "carrier_imu_source": gesture_timing.imu_source,
                    "carrier_imu_window": imu_audit,
                    "spread_policy": gesture_timing.spread_policy,
                    "law_log_offset": gesture_timing.log_offset,
                    "law_residual_draw": gesture_timing.residual_draw,
                    "law_residual_index": gesture_timing.residual_index,
                    "law_residual_spread": gesture_timing.residual_spread,
                    "floored_to_reportable": (
                        gesture_timing.floored_to_reportable
                    ),
                }
            if action == "keystroke":
                keystroke_plan = self.keystroke_target_plans.get(event_id)
                if (
                    keystroke_plan is None
                    or keystroke_plan.template.orientation_id
                    != target.orientation_id
                ):
                    raise ReplayDatasetBuildError(
                        "fake keystroke has no pre-bound replacement carrier"
                    )
                duration_sample = keystroke_plan.timing
                replay_raw_duration = keystroke_plan.raw_touch_duration_ms
                target_duration = keystroke_plan.detector_duration_ms
                target_samples = keystroke_plan.target_samples
                target_binding["superseded_bound_fake_target"] = {
                    "keycodes": list(target.keycodes),
                    "raw_touch_duration_ms": float(target.raw_duration_ms),
                    "detector_window_duration_ms": float(
                        input_target_duration
                    ),
                    "sample_count": int(input_samples),
                    "reason": (
                        "old_generated_keystroke_metadata_was_capped_to_one_"
                        "256_sample_diffusion_chunk"
                    ),
                }
                template = keystroke_plan.template
                target_binding["replacement_carrier"] = {
                    "source": "one_coupled_selected_train_genuine_event",
                    "source_cluster_id": template.source_cluster_id,
                    "source_event_id": template.source_event_id,
                    "source_user_id": template.source_user_id,
                    "source_session_id": template.source_session_id,
                    "orientation_id": template.orientation_id,
                    "keycodes": list(template.keycodes),
                    "down_anchors_px": [
                        list(value) for value in template.down_anchors_px
                    ],
                    "raw_touch_duration_ms": int(
                        keystroke_plan.raw_touch_duration_ms
                    ),
                    "detector_window_duration_ms": float(
                        keystroke_plan.detector_duration_ms
                    ),
                    "sample_count": int(keystroke_plan.target_samples),
                    "observable_update_count": int(
                        duration_sample.reference_observable_update_count
                    ),
                    "raw_trajectory_source": str(
                        template.raw_trajectory_source
                    ),
                    "raw_trajectory_source_sha256": (
                        template.raw_trajectory_source_sha256
                    ),
                    "raw_trajectory_event_index": (
                        template.raw_trajectory_event_index
                    ),
                    "raw_event_sha256": template.raw_event_sha256,
                    "old_fake_keycount_conditioning_used": False,
                }
                target_binding["duration_source"] = (
                    "coupled_selected_train_genuine_carrier"
                )
                target_binding["duration_ms"] = float(target_duration)
            elif (
                (
                    action in EXACT_TOUCH_TEMPLATE_ACTIONS
                    or action in PINCH_AREA_SIMILARITY_ACTIONS
                )
                and (
                    event_id in self.smoke_touch_templates
                    or event_id in self.material_assignment
                )
            ) or (
                action in CONDITIONAL_TOUCH_ACTIONS
                and self.conditional_touch_generator is not None
            ):
                duration_sample = None
                replay_raw_duration = target.raw_duration_ms
                target_binding["raw_duration_source"] = (
                    "bound_fake_android_target_timeline"
                )
            elif action == "tap" and self.tap_strategy == "bound_fake_tap_android_zoh":
                duration_sample = None
                replay_raw_duration = target.raw_duration_ms
                target_binding["raw_duration_source"] = (
                    "bound_fake_android_target_duration_ms"
                )
            else:
                duration_sample = self.duration_sampler.sample(
                    action=action,
                    orientation_id=target.orientation_id,
                    event_id=event_id,
                    seed=self.seed,
                    direction=target.direction,
                    pinch_scale_direction=target.pinch_scale_direction,
                    target_duration_ms=target_duration,
                    key_count=(
                        None
                    ),
                )
                replay_raw_duration = max(
                    1,
                    int(
                        round(
                            target_duration
                            * duration_sample.raw_to_window_ratio
                        )
                    ),
                )
            if duration_sample is not None:
                target_binding["sampled_raw_to_window_ratio"] = (
                    duration_sample.raw_to_window_ratio
                )
                target_binding["sampled_observable_update_rate_hz"] = (
                    duration_sample.target_update_rate_hz
                )
                target_binding["timing_reference_source_cluster_id"] = (
                    duration_sample.reference_source_cluster_id
                )
                target_binding["timing_reference_key_count"] = (
                    duration_sample.reference_key_count
                )
                target_binding["duration_ratio_conditioning"] = (
                    duration_sample.conditioning
                )
                target_binding["duration_ratio_conditioning_count"] = (
                    duration_sample.conditioning_count
                )
            target_binding["replay_raw_duration_ms"] = replay_raw_duration
            if action in EXACT_TOUCH_TEMPLATE_ACTIONS and (
                event_id in self.material_assignment
                or self.tap_strategy == "conditional_touch_generator"
            ):
                donor = self.material_assignment.get(event_id)
                if donor is not None:
                    # Five-shot route: the donor is one of this user's own five
                    # frozen events and the request is the attacker's bound
                    # Android target, so the transform actually runs instead of
                    # copying the carrier back to itself.
                    frozen_template = donor.trajectory
                    request_template = frozen_template
                    tap_drift_audit: dict[str, Any] = {}
                    start_px, end_px = _single_pointer_target_control_points_px(
                        target
                    )
                    requested_start_xy = start_px
                    requested_end_xy = end_px
                    requested_direction = target.direction
                    request_source = "frozen_fiveshot_material_exact_endpoints"
                    if action == "tap":
                        # The target reports one point for a tap, so honouring
                        # both of its rows would request a stationary gesture and
                        # discard the donor's own lift-off drift.
                        (
                            request_template,
                            requested_end_xy,
                            requested_direction,
                            tap_drift_audit,
                        ) = _fiveshot_tap_drift_request(
                            frozen_template, target, start_px=start_px
                        )
                        request_source = (
                            "frozen_fiveshot_material_donor_drift_endpoints"
                        )
                else:
                    frozen_template = self.smoke_touch_templates.get(event_id)
                    frozen_request = self.smoke_touch_requests.get(event_id)
                    if frozen_template is None or frozen_request is None:
                        raise ReplayDatasetBuildError(
                            "smoke tap/swipe requires an exact template and request"
                        )
                    request_template = frozen_template
                    tap_drift_audit = {}
                    (
                        requested_start_xy,
                        requested_end_xy,
                        requested_direction,
                    ) = frozen_request
                    request_source = "frozen_smoke_reference_exact_endpoints"
                if action in SYNTHESISED_CLOCK_ACTIONS and donor is not None:
                    observation, generation = _observe_synthesised_clock_touch(
                        request_template,
                        target,
                        target_samples=target_samples,
                        target_duration_ms=target_duration,
                        requested_start_xy=requested_start_xy,
                        requested_end_xy=requested_end_xy,
                        requested_direction=requested_direction,
                        seed=self.seed,
                    )
                else:
                    observation, generation = _observe_exact_touch_template(
                        request_template,
                        target,
                        target_samples=target_samples,
                        target_duration_ms=target_duration,
                        requested_start_xy=requested_start_xy,
                        requested_end_xy=requested_end_xy,
                        requested_direction=requested_direction,
                    )
                target_binding["gesture_endpoint_request_policy"] = request_source
                target_binding["gesture_requested_start_px"] = generation[
                    "requested_start_px"
                ]
                target_binding["gesture_requested_end_px"] = generation[
                    "requested_end_px"
                ]
                generator_source = Path(__file__).with_name(
                    "exact_touch_template_generator.py"
                ).resolve()
                donor = {
                    "role": "exact_touch_template_generator",
                    "human_replay": False,
                    "runtime_donor_used": True,
                    "model_used": False,
                    "source_template_event_id": (
                        event_id if donor is None else donor.event_id
                    ),
                    "source_template_sha256": hashlib.sha256(
                        np.ascontiguousarray(frozen_template).tobytes()
                    ).hexdigest(),
                    "source_material_shot_ordinal": (
                        None if donor is None else int(donor.shot_ordinal)
                    ),
                    "source_material_cluster_id": (
                        None if donor is None else donor.source_cluster_id
                    ),
                    "source_material_samples": (
                        None if donor is None else int(donor.samples)
                    ),
                    "template_resampled_to_carrier": (
                        None
                        if donor is None
                        else bool(int(donor.samples) != int(target_samples))
                    ),
                    "generator_schema_version": (
                        EXACT_TOUCH_TEMPLATE_SCHEMA_VERSION
                    ),
                    "generator_source": str(generator_source),
                    "generator_source_sha256": sha256_file(generator_source),
                    "coordinate_clipping_used": False,
                    "linear_touch_interpolation_used": False,
                    "request_source": request_source,
                    "request_plan": None,
                    **tap_drift_audit,
                    **generation,
                    "target_binding": target_binding,
                }
                method = EXACT_TOUCH_TEMPLATE_REBUILD_METHOD
            elif (
                action in PINCH_AREA_SIMILARITY_ACTIONS
                and event_id in self.material_assignment
            ):
                assigned = self.material_assignment[event_id]
                requested_area = _pinch_target_endpoint_geometry(target)
                observation, generation, material = self._place_pinch(
                    assigned,
                    target,
                    target_samples=target_samples,
                    target_duration_ms=target_duration,
                    requested=requested_area,
                    user_id=str(shard.arrays["user_id"][index]),
                )
                target_binding["gesture_endpoint_request_policy"] = (
                    "bound_fake_requested_pinch_area_attacker_chosen_magnitude"
                )
                target_binding["pinch_requested_start_points_px"] = [
                    list(value) for value in requested_area.start_points_px
                ]
                target_binding["pinch_requested_start_span_px"] = (
                    requested_area.start_span_px
                )
                target_binding["pinch_carrier_end_span_px_not_requested"] = (
                    requested_area.end_span_px
                )
                donor = {
                    "role": "fiveshot_pinch_area_similarity",
                    "human_replay": False,
                    "runtime_donor_used": True,
                    "model_used": False,
                    "source_template_event_id": material.event_id,
                    "source_template_sha256": hashlib.sha256(
                        np.ascontiguousarray(material.trajectory).tobytes()
                    ).hexdigest(),
                    "source_material_shot_ordinal": int(material.shot_ordinal),
                    "source_material_cluster_id": material.source_cluster_id,
                    "source_material_samples": int(material.samples),
                    "template_resampled_to_carrier": bool(
                        int(material.samples) != int(target_samples)
                    ),
                    "generator_source": str(Path(__file__).resolve()),
                    "generator_source_sha256": sha256_file(
                        Path(__file__).resolve()
                    ),
                    "coordinate_clipping_used": False,
                    "linear_touch_interpolation_used": False,
                    "request_source": "frozen_fiveshot_material_requested_area",
                    "request_plan": None,
                    **generation,
                    "target_binding": target_binding,
                }
                method = PINCH_AREA_SIMILARITY_REBUILD_METHOD
            elif (
                action in CONDITIONAL_TOUCH_ACTIONS
                and self.conditional_touch_generator is not None
            ):
                generator_seed = _event_seed(event_id, seed=self.seed)
                request_plan: ConditionalTouchRequestPlan | None = None
                requested_start_xy = None
                requested_end_xy = None
                requested_direction = None
                request_source = "bound_fake_exact_DOWN_UP_coordinates"
                if self.conditional_touch_request_generator is not None:
                    original_start, original_end = (
                        _single_pointer_target_control_points_px(target)
                    )
                    request_seed = _conditional_touch_request_seed(
                        event_id, seed=self.seed
                    )
                    try:
                        if target.direction is None:
                            raise ReplayDatasetBuildError(
                                "conditional scroll target lacks a direction"
                            )
                        sampled_request = (
                            self.conditional_touch_request_generator.generate(
                                action=action,
                                orientation_id=target.orientation_id,
                                direction=target.direction,
                                start_xy_px=original_start,
                                duration_ms=target.raw_duration_ms,
                                seed=request_seed,
                            )
                        )
                    except ConditionalTouchRequestGeneratorError as exc:
                        raise ReplayDatasetBuildError(
                            f"conditional touch request generation failed: {exc}"
                        ) from exc
                    requested_start_xy = sampled_request.start_xy_px
                    requested_end_xy = sampled_request.end_xy_px
                    requested_direction = (
                        None
                        if sampled_request.direction == "stationary"
                        else sampled_request.direction
                    )
                    request_plan = ConditionalTouchRequestPlan.create(
                        carrier_event_id=event_id,
                        original_event_plan_sha256=(
                            target.bound_event_plan_sha256
                        ),
                        orientation_id=target.orientation_id,
                        action=action,
                        original_direction=target.direction,
                        original_down_xy_px=original_start,
                        original_up_xy_px=original_end,
                        original_raw_t_ms=target.t_ms,
                        original_raw_duration_ms=target.raw_duration_ms,
                        sampled_start_xy_px=requested_start_xy,
                        sampled_end_xy_px=requested_end_xy,
                        sampled_direction=requested_direction,
                        request_model_file_sha256=(
                            self.conditional_touch_request_model_file_sha256
                        ),
                        request_model_artifact_sha256=(
                            self.conditional_touch_request_model_artifact_sha256
                        ),
                        request_model_schema_version=(
                            self.conditional_touch_request_generator.schema_version
                        ),
                        request_model_source_fingerprint_sha256=(
                            CONDITIONAL_TOUCH_REQUEST_SOURCE_FINGERPRINT_SHA256
                        ),
                        request_seed=request_seed,
                    )
                    request_source = "train_fitted_request_model"
                observation, generation = _observe_conditional_touch_target(
                    self.conditional_touch_generator,
                    target,
                    target_samples=target_samples,
                    target_duration_ms=target_duration,
                    generator_seed=generator_seed,
                    requested_start_xy=requested_start_xy,
                    requested_end_xy=requested_end_xy,
                    requested_direction=requested_direction,
                )
                target_binding["gesture_endpoint_request_policy"] = (
                    request_source
                )
                target_binding["gesture_requested_start_px"] = generation[
                    "requested_start_px"
                ]
                target_binding["gesture_requested_end_px"] = generation[
                    "requested_end_px"
                ]
                if request_plan is not None:
                    target_binding["identity_role"] = request_plan.identity_role
                    target_binding["plan_replay_semantics"] = (
                        request_plan.plan_replay_semantics
                    )
                    target_binding["original_carrier_request"] = {
                        "start_px": list(request_plan.original_down_xy_px),
                        "end_px": list(request_plan.original_up_xy_px),
                        "direction": request_plan.original_direction,
                    }
                    target_binding["sampled_request"] = {
                        "start_px": list(request_plan.sampled_start_xy_px),
                        "end_px": list(request_plan.sampled_end_xy_px),
                        "direction": request_plan.sampled_direction,
                        "request_plan_sha256": (
                            request_plan.request_plan_sha256
                        ),
                    }
                donor = {
                    "role": "frozen_conditional_touch_generator_model",
                    "human_replay": False,
                    "runtime_donor_used": False,
                    "model_used": True,
                    "source_template_event_id": None,
                    "model_schema_version": (
                        self.conditional_touch_generator.schema_version
                    ),
                    "model_artifact": str(self.conditional_touch_model_path),
                    "model_file_sha256": (
                        self.conditional_touch_model_file_sha256
                    ),
                    "model_canonical_artifact_sha256": (
                        self.conditional_touch_model_artifact_sha256
                    ),
                    "model_training_summary_sha256": (
                        self.conditional_touch_training_summary_sha256
                    ),
                    "coordinate_clipping_used": False,
                    "linear_touch_interpolation_used": False,
                    "request_source": request_source,
                    "request_plan": (
                        None
                        if request_plan is None
                        else request_plan.to_json_dict()
                    ),
                    "request_model_artifact": (
                        None
                        if self.conditional_touch_request_model_path is None
                        else str(self.conditional_touch_request_model_path)
                    ),
                    **generation,
                    "target_binding": target_binding,
                }
                method = CONDITIONAL_TOUCH_REBUILD_METHOD
            elif action in {"scroll", "swipe", "pinch"}:
                if target.direction is None:
                    raise ReplayDatasetBuildError("action target lacks a direction")
                allocator = self.action_allocators[(shard.source.split, action)]
                target_anchor = _action_target_anchor_px(target)
                target_update_count = (
                    float(duration_sample.reference_observable_update_count)
                    if duration_sample.reference_observable_update_count is not None
                    else None
                )
                if action in {"scroll", "swipe"}:
                    original_target_endpoint = _single_pointer_target_endpoint_px(
                        target
                    )
                    base_allocation = allocator.allocate_isometric_request(
                        orientation_id=target.orientation_id,
                        direction=target.direction,
                        detector_duration_ms=target_duration,
                        target_update_count=target_update_count,
                        target_update_rate_hz=(
                            None
                            if target_update_count is not None
                            else duration_sample.target_update_rate_hz
                        ),
                        target_anchor_px=target_anchor,
                    )
                    descriptor = base_allocation.descriptor
                    base_matrix = np.asarray(
                        base_allocation.isometry.matrix_xy, dtype=np.float64
                    )
                    base_translation = np.asarray(
                        base_allocation.isometry.translation_px, dtype=np.float64
                    )
                    endpoint_array = (
                        base_matrix
                        @ np.asarray(descriptor.end_xy_px, dtype=np.float64)
                        + base_translation
                    )
                    target_endpoint = (
                        float(endpoint_array[0]),
                        float(endpoint_array[1]),
                    )
                    allocation = IsometricReplayAllocation(
                        descriptor=descriptor,
                        isometry=replace(
                            base_allocation.isometry,
                            name=(
                                "endpoint_reference_"
                                + base_allocation.isometry.name
                            ),
                            requested_endpoint_px=target_endpoint,
                            output_endpoint_px=target_endpoint,
                            endpoint_error_px=0.0,
                        ),
                    )
                    geometry_reference_id = descriptor.primitive_id
                    geometry_reference_distance = float(
                        np.linalg.norm(endpoint_array - np.asarray(target_anchor))
                    )
                    target_binding["gesture_original_endpoint_px"] = list(
                        original_target_endpoint
                    )
                    target_binding["gesture_endpoint_request_policy"] = (
                        "selected_train_real_d4_exact_start_endpoint"
                    )
                    target_binding["gesture_geometry_reference_primitive_id"] = (
                        geometry_reference_id
                    )
                    target_binding["gesture_geometry_reference_distance_px"] = (
                        geometry_reference_distance
                    )
                else:
                    target_endpoint = None
                    pinch_target_geometry = _pinch_target_endpoint_geometry(
                        target
                    )
                    pinch_allocation = allocator.allocate_pinch_endpoint_request(
                        orientation_id=target.orientation_id,
                        direction=target.direction,
                        detector_duration_ms=target_duration,
                        pinch_scale_direction=str(
                            target.pinch_scale_direction
                        ),
                        target_geometry=pinch_target_geometry,
                        target_update_count=target_update_count,
                        target_update_rate_hz=(
                            None
                            if target_update_count is not None
                            else duration_sample.target_update_rate_hz
                        ),
                    )
                    descriptor = pinch_allocation.descriptor
                    target_binding["gesture_endpoint_request_policy"] = (
                        "bound_fake_exact_live_pointer_endpoints_with_"
                        "train_real_bounded_donor"
                    )
                    target_binding["pinch_requested_start_points_px"] = [
                        list(value)
                        for value in pinch_target_geometry.start_points_px
                    ]
                    target_binding["pinch_requested_end_points_px"] = [
                        list(value)
                        for value in pinch_target_geometry.end_points_px
                    ]
                    target_binding["pinch_requested_start_span_px"] = (
                        pinch_target_geometry.start_span_px
                    )
                    target_binding["pinch_requested_end_span_px"] = (
                        pinch_target_geometry.end_span_px
                    )
                if action in {"scroll", "swipe"}:
                    descriptor = allocation.descriptor
                # Action contacts remain on their original integer-ms lattice.
                # The sampled human timing reference conditions donor update
                # density in the detector window; it never stretches raw time.
                target_binding["raw_ratio_implied_duration_ms_not_applied"] = (
                    replay_raw_duration
                )
                replay_raw_duration = float(descriptor.duration_ms)
                target_binding["replay_raw_duration_ms"] = replay_raw_duration
                target_binding["replay_raw_duration_source"] = (
                    "selected_train_raw_donor_unchanged"
                )
                target_binding["gesture_start_anchor_px"] = list(target_anchor)
                if target_endpoint is not None:
                    target_binding["gesture_requested_endpoint_px"] = list(target_endpoint)
                if action == "pinch":
                    replay = observe_replay_primitive(
                        self.action_banks[action],
                        descriptor.primitive_id,
                        target_samples=input_samples,
                        replay_duration_ms=replay_raw_duration,
                        output_duration_ms=target_duration,
                        pinch_endpoint_fit=pinch_allocation.endpoint_fit,
                        max_time_warp=1.0,
                    )
                    try:
                        pinch_output_geometry = (
                            extract_live_two_pointer_endpoints(
                                t_ms=replay.rows.t_ms,
                                frame_index=replay.rows.frame_index,
                                pointer_id=replay.rows.pointer_id,
                                android_action=replay.rows.android_action,
                                x_px=replay.rows.x_px,
                                y_px=replay.rows.y_px,
                            )
                        )
                    except PinchEndpointControlError as error:
                        raise ReplayDatasetBuildError(
                            "replayed pinch endpoint audit failed: "
                            f"{error}"
                        ) from error
                    pinch_start_endpoint_error = float(
                        np.max(
                            np.linalg.norm(
                                np.asarray(
                                    pinch_output_geometry.start_points_px
                                )
                                - np.asarray(
                                    pinch_allocation.endpoint_fit.target
                                    .start_points_px
                                ),
                                axis=1,
                            )
                        )
                    )
                    pinch_end_endpoint_error = float(
                        np.max(
                            np.linalg.norm(
                                np.asarray(
                                    pinch_output_geometry.end_points_px
                                )
                                - np.asarray(
                                    pinch_allocation.endpoint_fit.target
                                    .end_points_px
                                ),
                                axis=1,
                            )
                        )
                    )
                else:
                    replay = observe_replay_primitive(
                        self.action_banks[action],
                        descriptor.primitive_id,
                        target_samples=input_samples,
                        replay_duration_ms=replay_raw_duration,
                        output_duration_ms=target_duration,
                        spatial_isometry=allocation.isometry,
                        max_time_warp=1.0,
                    )
                observation = replay.observation
                if action == "pinch":
                    endpoint_fit = pinch_allocation.endpoint_fit
                    scale_used = any(
                        not math.isclose(value, 1.0, abs_tol=1.0e-12)
                        for value in (
                            endpoint_fit.center_scale,
                            endpoint_fit.start_span_scale,
                            endpoint_fit.end_span_scale,
                        )
                    )
                    spatial_scale = endpoint_fit.center_scale
                    spatial_transform_name = (
                        "pinch_bounded_endpoint_residual"
                    )
                    spatial_matrix_xy = None
                    requested_anchor_px = endpoint_fit.target.start_center_px
                    output_anchor_px = endpoint_fit.target.start_center_px
                    anchor_error_px = 0.0
                    requested_endpoint_px = endpoint_fit.target.end_center_px
                    output_endpoint_px = endpoint_fit.target.end_center_px
                    endpoint_error_px = 0.0
                    requested_distance_ratio = endpoint_fit.center_scale
                    endpoint_residual_px = None
                    endpoint_residual_fraction = None
                    pixel_lattice_correction_used = False
                    spatial_transform_rank = None
                else:
                    scale_used = not math.isclose(
                        allocation.isometry.spatial_scale,
                        1.0,
                        abs_tol=1.0e-12,
                    )
                    spatial_scale = allocation.isometry.spatial_scale
                    spatial_transform_name = allocation.isometry.name
                    spatial_matrix_xy = [
                        list(row) for row in allocation.isometry.matrix_xy
                    ]
                    requested_anchor_px = (
                        allocation.isometry.requested_anchor_px
                    )
                    output_anchor_px = allocation.isometry.output_anchor_px
                    anchor_error_px = allocation.isometry.anchor_error_px
                    requested_endpoint_px = (
                        allocation.isometry.requested_endpoint_px
                    )
                    output_endpoint_px = allocation.isometry.output_endpoint_px
                    endpoint_error_px = allocation.isometry.endpoint_error_px
                    requested_distance_ratio = (
                        allocation.isometry.requested_distance_ratio
                    )
                    endpoint_residual_px = (
                        allocation.isometry.endpoint_residual_px
                    )
                    endpoint_residual_fraction = (
                        allocation.isometry.endpoint_residual_fraction
                    )
                    pixel_lattice_correction_used = (
                        allocation.isometry.quantize_pixel_lattice
                    )
                    spatial_transform_rank = allocation.isometry.rank
                donor = {
                    "role": "train_raw_action_isometric_replay",
                    "primitive_id": descriptor.primitive_id,
                    "source_event_id": descriptor.source_event_id,
                    "source_user_id": descriptor.source_user_id,
                    "source_session_id": descriptor.source_session_id,
                    "source_event_index": descriptor.source_event_index,
                    "source_duration_ms": descriptor.duration_ms,
                    "source_observable_update_count": (
                        descriptor.observable_update_count
                    ),
                    "source_observable_update_rate_hz": (
                        descriptor.observable_update_rate_hz
                    ),
                    "requested_observable_update_rate_hz": (
                        duration_sample.target_update_rate_hz
                    ),
                    "requested_observable_update_count": (
                        duration_sample.reference_observable_update_count
                    ),
                    "detector_window_observable_update_rate_hz": (
                        descriptor.observable_update_count
                        * 1000.0
                        / float(target_duration)
                    ),
                    "replay_raw_duration_ms": replay.rows.replay_duration_ms,
                    "detector_window_duration_ms": replay.detector_duration_ms,
                    "orientation_id": descriptor.orientation_id,
                    "direction": descriptor.bucket.direction,
                    "source_direction": descriptor.bucket.direction,
                    "target_direction": target.direction,
                    "duration_bucket": descriptor.bucket.duration_bucket,
                    "pinch_scale_direction": descriptor.bucket.pinch_scale_direction,
                    "time_warp_ratio": replay.rows.time_warp_ratio,
                    "scale_used": scale_used,
                    "spatial_scale": spatial_scale,
                    "coordinate_clipping_used": False,
                    "spatial_transform_name": spatial_transform_name,
                    "spatial_matrix_xy": spatial_matrix_xy,
                    "translation_px": list(replay.rows.translation_px),
                    "requested_anchor_px": list(requested_anchor_px),
                    "output_anchor_px": list(output_anchor_px),
                    "anchor_error_px": anchor_error_px,
                    "requested_endpoint_px": (
                        None
                        if requested_endpoint_px is None
                        else list(requested_endpoint_px)
                    ),
                    "output_endpoint_px": (
                        None
                        if output_endpoint_px is None
                        else list(output_endpoint_px)
                    ),
                    "endpoint_error_px": endpoint_error_px,
                    "requested_distance_ratio": requested_distance_ratio,
                    "endpoint_residual_px": (
                        None
                        if endpoint_residual_px is None
                        else list(endpoint_residual_px)
                    ),
                    "endpoint_residual_fraction": endpoint_residual_fraction,
                    "pixel_lattice_correction_used": (
                        pixel_lattice_correction_used
                    ),
                    "spatial_transform_rank": spatial_transform_rank,
                    "raw_archive": str(self.action_banks[action].source_path),
                    "raw_archive_sha256": self.action_banks[action].source_sha256,
                    "target_binding": target_binding,
                }
                if action == "pinch":
                    donor.update(
                        {
                            "role": "train_raw_action_pinch_endpoint_replay",
                            "pinch_source_start_points_px": [
                                list(value)
                                for value in endpoint_fit.source.start_points_px
                            ],
                            "pinch_source_end_points_px": [
                                list(value)
                                for value in endpoint_fit.source.end_points_px
                            ],
                            "pinch_requested_start_points_px": [
                                list(value)
                                for value in endpoint_fit.target.start_points_px
                            ],
                            "pinch_requested_end_points_px": [
                                list(value)
                                for value in endpoint_fit.target.end_points_px
                            ],
                            "pinch_center_scale": endpoint_fit.center_scale,
                            "pinch_start_span_scale": (
                                endpoint_fit.start_span_scale
                            ),
                            "pinch_end_span_scale": endpoint_fit.end_span_scale,
                            "pinch_center_rotation_rad": (
                                endpoint_fit.center_rotation_rad
                            ),
                            "pinch_deformation_score": (
                                endpoint_fit.deformation_score
                            ),
                            "pinch_minimum_scale": endpoint_fit.minimum_scale,
                            "pinch_maximum_scale": endpoint_fit.maximum_scale,
                            "pinch_start_endpoint_error_px": (
                                pinch_start_endpoint_error
                            ),
                            "pinch_end_endpoint_error_px": (
                                pinch_end_endpoint_error
                            ),
                            "pinch_maximum_endpoint_error_px": max(
                                pinch_start_endpoint_error,
                                pinch_end_endpoint_error,
                            ),
                        }
                    )
                    method = "train_raw_action_pinch_endpoint_replay"
                else:
                    method = "train_raw_action_isometric_replay"
            elif action == "keystroke":
                keystroke_rhythm = self.keystroke_rhythm(
                    str(shard.arrays["user_id"][index])
                )
                if keystroke_rhythm is None:
                    raise ReplayDatasetBuildError(
                        "keystroke generation requires five-shot material"
                    )
                template = keystroke_plan.template
                width_px, height_px = screen_dimensions_for_orientation(
                    target.orientation_id
                )
                try:
                    composed_touch = compose_keystroke_touch(
                        anchors_px=template.down_anchors_px,
                        rhythm=keystroke_rhythm,
                        target_samples=target_samples,
                        target_duration_ms=float(target_duration),
                        screen_width_px=float(width_px),
                        screen_height_px=float(height_px),
                        bounds=ROBUST_KEYSTROKE_BOUNDS,
                        seed=_event_seed(event_id, seed=self.seed),
                    )
                except FiveShotKeystrokeTouchError as exc:
                    raise ReplayDatasetBuildError(
                        f"five-shot keystroke touch failed: {exc}"
                    ) from exc
                trajectory = composed_touch.trajectory
                observation = TouchObservation(
                    touch=trajectory[:, :7].copy(),
                    trajectory=trajectory.copy(),
                    touch_observed=bool(np.any(trajectory[:, 8] > 0.5)),
                    source_updates=int(len(trajectory)),
                )
                pulse_model = self.keystroke_pulse_model(
                    str(shard.arrays["user_id"][index])
                )
                if pulse_model is None:
                    raise ReplayDatasetBuildError(
                        "five-shot keystroke needs its victim's IMU adapter"
                    )
                # The IMU is driven from the schedule the contacts were placed
                # on, so each impact lands at its own key press rather than at a
                # separately sampled time.
                letters = [
                    bool(65 <= int(code) <= 90 or 97 <= int(code) <= 122)
                    for code in template.keycodes
                ]
                try:
                    generated_imu = generate_keystroke_imu(
                        pulse_model,
                        key_down_ms=composed_touch.key_down_ms,
                        is_letter=letters,
                        duration_ms=float(target_duration),
                        seed=_event_seed(event_id, seed=self.seed),
                        event_id=event_id,
                    )
                except KeystrokeImuPulseError as exc:
                    raise ReplayDatasetBuildError(
                        f"five-shot keystroke IMU generation failed: {exc}"
                    ) from exc
                output_imu = _match_keystroke_imu_samples(
                    generated_imu.imu, target_samples
                )
                generator_source = Path(__file__).with_name(
                    "fiveshot_keystroke_touch.py"
                ).resolve()
                pulse_source = Path(__file__).with_name(
                    "keystroke_imu_pulse.py"
                ).resolve()
                donor = {
                    "role": "fiveshot_keystroke_rhythm_touch",
                    "human_replay": False,
                    "runtime_donor_used": True,
                    "model_used": False,
                    "material_user_id": str(shard.arrays["user_id"][index]),
                    "material_source_event_ids": list(
                        keystroke_rhythm.source_event_ids
                    ),
                    "material_hold_count": int(len(keystroke_rhythm.holds_ms)),
                    "material_flight_count": int(len(keystroke_rhythm.flights_ms)),
                    "target_keycodes": list(template.keycodes),
                    "target_down_anchors_px": [
                        list(value) for value in template.down_anchors_px
                    ],
                    "hold_ms": composed_touch.holds_ms.astype(int).tolist(),
                    "flight_ms": composed_touch.flights_ms.astype(int).tolist(),
                    "timing_bounds_ms": {
                        "hold": [
                            ROBUST_KEYSTROKE_BOUNDS.hold_min_ms,
                            ROBUST_KEYSTROKE_BOUNDS.hold_max_ms,
                        ],
                        "flight": [
                            ROBUST_KEYSTROKE_BOUNDS.flight_min_ms,
                            ROBUST_KEYSTROKE_BOUNDS.flight_max_ms,
                        ],
                    },
                    "contact_segments": int(composed_touch.contact_segments),
                    "contact_samples": int(composed_touch.contact_samples),
                    # Two keys pressed less than two detector periods apart land
                    # on adjacent samples and read as one contact.  That is a
                    # property of the observation grid, not of this generator:
                    # 4.6% of genuine HMOG typing events resolve fewer segments
                    # than they have keys, against 0.65% here.  The count is
                    # recorded so the merge is stated rather than silent.
                    "merged_contact_segments": (
                        len(template.keycodes)
                        - int(composed_touch.contact_segments)
                    ),
                    "coordinate_clipping_used": False,
                    "linear_touch_interpolation_used": False,
                    "keycode_matched_source_contact_required": False,
                    "generator_source": str(generator_source),
                    "generator_source_sha256": sha256_file(generator_source),
                    "generation_mode": FIVESHOT_KEYSTROKE_GENERATION_MODE,
                    "imu": {
                        "role": "fiveshot_keystroke_impulse_response_adapter",
                        "generator_source": str(pulse_source),
                        "generator_source_sha256": sha256_file(pulse_source),
                        "model_user_id": pulse_model.user_id,
                        "model_source_event_ids": list(
                            pulse_model.source_event_ids
                        ),
                        "explicit_timeline_sources": int(
                            pulse_model.explicit_timeline_sources
                        ),
                        "natural_samples": int(len(generated_imu.imu)),
                        "carrier_samples": int(target_samples),
                        "resampled_to_carrier": bool(
                            len(generated_imu.imu) != int(target_samples)
                        ),
                        **generated_imu.provenance,
                    },
                    "target_binding": target_binding,
                }
                method = FIVESHOT_KEYSTROKE_TOUCH_METHOD
            elif action == "tap" and self.tap_strategy == "train_raw_tap_replay":
                tap_allocator = self.tap_allocators[shard.source.split]
                tap_donor = tap_allocator.allocate(
                    orientation_id=target.orientation_id,
                    target_duration_ms=replay_raw_duration,
                )
                binding = tap_donor.binding
                observation = observe_genuine_binding(
                    binding,
                    target_samples=input_samples,
                    target_duration_ms=target_duration,
                    archive=_raw_archive_cached(binding, self.raw_archive_cache),
                )
                donor = {
                    "role": "train_raw_tap_replay",
                    "source_event_id": binding.source_event_id,
                    "source_user_id": binding.user_id,
                    "source_session_id": binding.session_id,
                    "source_raw_duration_ms": tap_donor.raw_duration_ms,
                    "orientation_id": binding.orientation_id,
                    "raw_trajectory_source": str(binding.raw_trajectory_source),
                    "raw_trajectory_source_sha256": binding.raw_trajectory_source_sha256,
                    "raw_trajectory_event_index": binding.raw_trajectory_event_index,
                    "raw_event_sha256": binding.raw_event_sha256,
                    "raw_duration_match_ratio": (
                        replay_raw_duration / tap_donor.raw_duration_ms
                    ),
                    "target_binding": target_binding,
                }
                method = "train_raw_tap_replay"
            elif action == "tap":
                observation = observe_bound_android_target(
                    target,
                    target_samples=target_samples,
                    target_duration_ms=target_duration,
                )
                donor = {
                    "role": "retained_bound_fake_tap_android_target",
                    "human_replay": False,
                    "trajectory_source": str(target.trajectory_source),
                    "trajectory_source_sha256": target.trajectory_source_sha256,
                    "trajectory_archive_index": target.trajectory_archive_index,
                    "source_raw_duration_ms": target.raw_duration_ms,
                    "detector_window_duration_ms": target_duration,
                    "orientation_id": target.orientation_id,
                    "target_binding": target_binding,
                }
                method = "bound_fake_tap_android_zoh"
            else:
                raise ReplayDatasetBuildError(f"unsupported action {action}")
        output_trajectory = np.asarray(observation.trajectory, dtype=np.float32)
        if (
            output_imu.ndim != 2
            or output_imu.shape[1] != 6
            or output_trajectory.shape != (len(output_imu), 9)
            or len(output_imu) < 2
        ):
            raise ReplayDatasetBuildError("rebuilt paired signal shape is invalid")
        if (len(output_imu) != input_samples) and not (
            label == 1
            and (
                action == "keystroke"
                or (
                    action in FIVESHOT_TIMING_ACTIONS
                    and event_id in self.fiveshot_timing
                )
            )
        ):
            raise ReplayDatasetBuildError(
                "only a fake keystroke or a fake gesture timed from the victim's "
                "material may change its carrier sample count"
            )
        if not np.isfinite(output_imu).all() or not np.isfinite(
            output_trajectory
        ).all():
            raise ReplayDatasetBuildError("rebuilt paired signal is non-finite")
        provenance = _output_event_provenance(
            shard=shard,
            index=index,
            input_imu=imu,
            output_imu=output_imu,
            old_trajectory=old_trajectory,
            observation=observation,
            method=method,
            donor=donor,
        )
        provenance["input_duration_ms"] = float(input_duration_ms)
        return RebuiltEventSignal(
            imu=output_imu.copy(), trajectory=output_trajectory.copy()
        ), provenance


def _rebuild_reference_locked_smoke_shard(
    *,
    context: ReplayContext,
    shard: LoadedShard,
    selection: Mapping[SmokeReferenceKey, Sequence[str]],
    record_provenance: Any,
) -> tuple[
    list[int],
    dict[int, RebuiltEventSignal],
    dict[SmokeReferenceKey, tuple[str, ...]],
]:
    """Rebuild only frozen smoke IDs, failing before any replacement attempt."""

    split = shard.source.split
    user_id = shard.source.user_id
    labels = np.asarray(shard.arrays["label"], dtype=np.int64)
    actions = np.asarray(shard.arrays["action"]).astype(str)
    event_ids = np.asarray(shard.arrays["event_id"]).astype(str)
    if not (len(labels) == len(actions) == len(event_ids) == shard.event_count):
        raise ReplayDatasetBuildError(
            "reference-locked smoke shard metadata is malformed"
        )
    index_by_event_id = {
        str(event_id): int(index)
        for index, event_id in enumerate(event_ids)
    }
    if len(index_by_event_id) != len(event_ids):
        raise ReplayDatasetBuildError(
            "reference-locked smoke shard repeats an event ID"
        )

    selected: list[int] = []
    signals: dict[int, RebuiltEventSignal] = {}
    observed: dict[SmokeReferenceKey, tuple[str, ...]] = {}
    for action in ACTIONS:
        for label in (0, 1):
            key = (split, user_id, action, label)
            frozen_ids = selection.get(key)
            if frozen_ids is None:
                raise ReplayDatasetBuildError(
                    f"smoke reference has no frozen group for {key}"
                )
            rebuilt_ids: list[str] = []
            for event_id in frozen_ids:
                identity = str(event_id)
                index = index_by_event_id.get(identity)
                if index is None or (
                    str(actions[index]) != action
                    or int(labels[index]) != label
                ):
                    raise ReplayDatasetBuildError(
                        "smoke reference event binding changed before rebuild: "
                        f"{identity}"
                    )
                try:
                    signal, provenance = context.rebuild(
                        shard=shard, index=index
                    )
                except (
                    ActionReplayError,
                    GenuineTouchRecoveryError,
                    KeystrokeReplayError,
                    ReplayDatasetBuildError,
                ) as exc:
                    raise ReplayDatasetBuildError(
                        "frozen smoke reference event failed rebuild; replacement "
                        "is forbidden: "
                        f"{split}/{user_id}/{action}/{label}/{identity}: {exc}"
                    ) from exc
                if (
                    str(provenance.get("event_id", "")) != identity
                    or str(provenance.get("split", "")) != split
                    or str(provenance.get("user_id", "")) != user_id
                    or str(provenance.get("action", "")) != action
                    or int(provenance.get("label", -1)) != label
                ):
                    raise ReplayDatasetBuildError(
                        "reference-locked rebuild provenance changed event binding"
                    )
                selected.append(index)
                signals[index] = signal
                record_provenance(provenance)
                rebuilt_ids.append(identity)
            observed[key] = tuple(rebuilt_ids)
    expected_count = sum(len(values) for values in observed.values())
    if (
        len(selected) != expected_count
        or len(signals) != expected_count
        or len(selected) != len(set(selected))
    ):
        raise ReplayDatasetBuildError(
            "reference-locked smoke shard rebuild coverage is incomplete"
        )
    return selected, signals, observed


def _finalize_smoke_reference_audit(
    *,
    expected: Mapping[SmokeReferenceKey, Sequence[str]],
    observed: Mapping[SmokeReferenceKey, Sequence[str]],
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    normalized_expected = {
        key: tuple(str(value) for value in values)
        for key, values in expected.items()
    }
    normalized_observed = {
        key: tuple(str(value) for value in values)
        for key, values in observed.items()
    }
    if normalized_observed != normalized_expected:
        raise ReplayDatasetBuildError(
            "output smoke event IDs differ from the frozen reference selection"
        )
    output_digest = _smoke_reference_selection_sha256(normalized_observed)
    if output_digest != str(audit.get("frozen_event_ids_sha256", "")):
        raise ReplayDatasetBuildError(
            "output smoke reference selection digest changed"
        )
    result = dict(audit)
    result.update(
        {
            "status": "pass",
            "rebuilt_events": sum(
                len(values) for values in normalized_observed.values()
            ),
            "output_event_ids_sha256": output_digest,
            "output_event_ids_exact_match": True,
            "replacement_after_rebuild_failure_used": False,
        }
    )
    return result


def _write_output_shard(
    *,
    output_path: Path,
    shard: LoadedShard,
    selected: Sequence[int],
    signals: Mapping[int, RebuiltEventSignal],
    scope: str = SMOKE_MANIFEST_SCOPE,
) -> dict[str, Any]:
    requested = [int(value) for value in selected]
    indices = np.asarray(sorted(requested), dtype=np.int64)
    if len(indices) != len(set(indices.tolist())):
        raise ReplayDatasetBuildError("output event indices are duplicated")
    if np.any(indices < 0) or np.any(indices >= shard.event_count):
        raise ReplayDatasetBuildError("output event index is out of range")
    signal_indices = {int(value) for value in signals}
    if signal_indices != set(indices.tolist()):
        raise ReplayDatasetBuildError(
            "rebuilt signal keys do not exactly match selected event indices"
        )
    imu_parts: list[np.ndarray] = []
    trajectory_parts: list[np.ndarray] = []
    pair_ids: list[str] = []
    offsets = [0]
    for index in indices:
        signal = signals[int(index)]
        imu = np.asarray(signal.imu, dtype=np.float32)
        trajectory = np.asarray(signal.trajectory, dtype=np.float32)
        if (
            imu.ndim != 2
            or imu.shape[1] != 6
            or trajectory.shape != (len(imu), 9)
            or len(imu) < 2
            or not np.isfinite(imu).all()
            or not np.isfinite(trajectory).all()
        ):
            raise ReplayDatasetBuildError("output paired event shape is invalid")
        imu_parts.append(imu.copy())
        trajectory_parts.append(trajectory.copy())
        offsets.append(offsets[-1] + len(imu))
        pair_ids.append(
            _pair_id(str(shard.arrays["event_id"][index]), imu, trajectory)
        )
    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray(INPUT_SHARD_SCHEMA),
        "coordinate_schema": np.asarray(
            str(np.asarray(shard.arrays["coordinate_schema"]).item())
        ),
        "time_schema": np.asarray(str(np.asarray(shard.arrays["time_schema"]).item())),
        "scope": np.asarray(scope),
        "split": np.asarray(shard.source.split),
        "imu_flat": np.concatenate(imu_parts).astype(np.float32, copy=False),
        "trajectory_flat": np.concatenate(trajectory_parts).astype(
            np.float32, copy=False
        ),
        "offsets": np.asarray(offsets, dtype=np.int64),
        "cross_modal_pair_id": np.asarray(pair_ids),
    }
    for name in (
        "label",
        "user_id",
        "session_id",
        "event_id",
        "action",
        "source_cluster_id",
        "sample_idx",
    ):
        arrays[name] = np.asarray(shard.arrays[name][indices]).copy()
    np.savez_compressed(output_path, **arrays)
    if not output_path.is_file():
        raise ReplayDatasetBuildError(f"failed to write output shard {output_path}")
    _validate_written_shard_pair_bindings(output_path)
    labels = np.asarray(arrays["label"], dtype=np.int64)
    actions = np.asarray(arrays["action"]).astype(str)
    return {
        "source": str(output_path.resolve()),
        "source_sha256": sha256_file(output_path),
        "user_id": shard.source.user_id,
        "events": int(len(indices)),
        "genuine": int(np.sum(labels == 0)),
        "fake": int(np.sum(labels == 1)),
        "actions": sorted(set(actions.tolist()), key=ACTIONS.index),
    }


def _validate_written_shard_pair_bindings(path: Path) -> None:
    """Recompute every pair binding from the bytes actually written to disk."""

    source = path.resolve()
    required = {
        "event_id",
        "imu_flat",
        "trajectory_flat",
        "offsets",
        "cross_modal_pair_id",
    }
    with np.load(source, allow_pickle=False) as archive:
        missing = required - set(archive.files)
        if missing:
            raise ReplayDatasetBuildError(
                f"written shard is missing pair arrays: {sorted(missing)}"
            )
        event_ids = np.asarray(archive["event_id"]).astype(str)
        imu_flat = np.asarray(archive["imu_flat"], dtype=np.float32)
        trajectory_flat = np.asarray(
            archive["trajectory_flat"], dtype=np.float32
        )
        offsets = np.asarray(archive["offsets"], dtype=np.int64)
        declared = np.asarray(archive["cross_modal_pair_id"]).astype(str)
    if (
        offsets.shape != (len(event_ids) + 1,)
        or declared.shape != event_ids.shape
        or int(offsets[0]) != 0
        or np.any(np.diff(offsets) < 2)
        or int(offsets[-1]) != len(imu_flat)
        or len(trajectory_flat) != len(imu_flat)
        or imu_flat.shape[1:] != (6,)
        or trajectory_flat.shape[1:] != (9,)
    ):
        raise ReplayDatasetBuildError("written shard ragged pair contract changed")
    observed: list[str] = []
    for index, event_id in enumerate(event_ids):
        left, right = (int(value) for value in offsets[index : index + 2])
        observed.append(
            _pair_id(
                str(event_id),
                imu_flat[left:right],
                trajectory_flat[left:right],
            )
        )
    if not np.array_equal(np.asarray(observed), declared):
        raise ReplayDatasetBuildError(
            "written shard cross-modal pair binding does not match its signals"
        )


def _provenance_native_int(
    value: Any, *, name: str, minimum: int | None = None
) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(
        value, (bool, np.bool_)
    ):
        raise ReplayDatasetBuildError(f"{name} must be an integer")
    output = int(value)
    if minimum is not None and output < minimum:
        raise ReplayDatasetBuildError(f"{name} is below its frozen minimum")
    return output


def _provenance_finite_number(value: Any, *, name: str) -> float:
    if not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(
        value, (bool, np.bool_)
    ):
        raise ReplayDatasetBuildError(f"{name} must be a finite number")
    output = float(value)
    if not math.isfinite(output):
        raise ReplayDatasetBuildError(f"{name} must be a finite number")
    return output


def _exact_touch_imu_provenance_is_valid(row: Mapping[str, Any]) -> bool:
    """Check the inertia a rebuilt gesture kept, or the window it was cut from."""

    donor = row.get("donor") or {}
    binding = donor.get("target_binding") or {}
    timing = binding.get("fiveshot_gesture_timing")
    if timing is None:
        return row.get("input_imu_sha256") == row.get("output_imu_sha256")
    window = timing.get("carrier_imu_window") or {}
    span = window.get("cut_span")
    active = window.get("carrier_active_span")
    if not isinstance(span, list) or not isinstance(active, list):
        return False
    if len(span) != 2 or len(active) != 2:
        return False
    window_samples = window.get("carrier_window_samples")
    if not isinstance(window_samples, int) or window_samples < 2:
        return False
    if not 0 <= span[0] < span[1] <= window_samples:
        return False
    if not 0 <= active[0] < active[1] <= window_samples:
        return False
    if int(row.get("output_samples", -1)) != span[1] - span[0]:
        return False
    if not isinstance(timing.get("carrier_imu_source"), str):
        return False
    travel = timing.get("requested_travel_px")
    if not isinstance(travel, float) or not math.isfinite(travel) or travel < 0.0:
        return False
    sources = timing.get("law_source_event_ids")
    if not isinstance(sources, list) or len(sources) < 2:
        return False
    capped = timing.get("capped_to_carrier_window")
    if not isinstance(capped, bool):
        return False
    requested = timing.get("requested_samples")
    if not isinstance(requested, int) or requested < 2:
        return False
    if capped != (requested != span[1] - span[0]):
        return False
    policy = timing.get("spread_policy")
    if policy not in FIVESHOT_TIMING_SPREAD_POLICIES:
        return False
    offset = timing.get("law_log_offset")
    if not isinstance(offset, float) or not math.isfinite(offset):
        return False
    # A curve-only build must not have moved the reading; a spread build must
    # have drawn its departure from the victim's own material, so a zero offset
    # there can only come from a victim whose recordings cannot be held out.
    if policy == "none" and offset != 0.0:
        return False
    spread = timing.get("law_residual_spread")
    if not isinstance(spread, float) or not math.isfinite(spread) or spread < 0.0:
        return False
    floored = timing.get("floored_to_reportable")
    if not isinstance(floored, bool):
        return False
    index = timing.get("law_residual_index")
    if not isinstance(index, int) or index < -1:
        return False
    # The curve alone always reads between two of the victim's own recordings,
    # so it can neither draw a departure nor ask for a gesture too short to
    # report; only a spread build can do either.
    return not (policy == "none" and (floored or index != -1))


def _validate_donor_provenance(
    rows: Iterable[Mapping[str, Any]],
    *,
    selected_genuine_source_event_ids: set[str],
) -> dict[str, int]:
    primitive_by_split = {split: set() for split in SPLITS}
    tap_by_split = {split: set() for split in SPLITS}
    all_donor_events: set[str] = set()
    exact_touch_generated_events = 0
    pinch_area_similarity_events = 0
    fiveshot_keystroke_events = 0
    exact_touch_generator_digest_by_path: dict[Path, str] = {}
    conditional_touch_generated_events = 0
    conditional_touch_model_digest_by_path: dict[Path, str] = {}
    for row in rows:
        if int(row["label"]) != 1:
            continue
        split = str(row["split"])
        method = str(row["rebuild_method"])
        donor = row["donor"]
        if method == EXACT_TOUCH_TEMPLATE_REBUILD_METHOD:
            action = str(row.get("action", ""))
            try:
                requested = np.asarray(
                    (donor["requested_start_px"], donor["requested_end_px"]),
                    dtype=np.float64,
                )
                raw_output = np.asarray(
                    (donor["raw_output_start_px"], donor["raw_output_end_px"]),
                    dtype=np.float64,
                )
                detector_output = np.asarray(
                    (
                        donor["detector_output_start_px"],
                        donor["detector_output_end_px"],
                    ),
                    dtype=np.float64,
                )
                declared_errors = np.asarray(
                    (
                        donor["raw_start_error_px"],
                        donor["raw_end_error_px"],
                        donor["detector_start_error_px"],
                        donor["detector_end_error_px"],
                    ),
                    dtype=np.float64,
                )
                requested_rows = _provenance_native_int(
                    donor["requested_raw_row_count"],
                    name="exact touch requested rows",
                )
                generated_rows = _provenance_native_int(
                    donor["generated_raw_row_count"],
                    name="exact touch generated rows",
                )
                requested_duration = _provenance_finite_number(
                    donor["requested_raw_duration_ms"],
                    name="exact touch requested duration",
                )
                generated_duration = _provenance_finite_number(
                    donor["generated_raw_duration_ms"],
                    name="exact touch generated duration",
                )
                residual_scale = _provenance_finite_number(
                    donor["residual_scale"], name="exact touch residual scale"
                )
                generator_path = Path(str(donor["generator_source"])).resolve()
                generator_digest = str(donor["generator_source_sha256"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ReplayDatasetBuildError(
                    "exact touch template provenance is incomplete"
                ) from exc
            raw_errors = np.linalg.norm(raw_output - requested, axis=1)
            detector_errors = np.linalg.norm(
                detector_output - requested, axis=1
            )
            recomputed_errors = np.concatenate((raw_errors, detector_errors))
            maximum_error = float(np.max(recomputed_errors))
            forbidden_model_fields = {
                "model_artifact",
                "model_file_sha256",
                "model_canonical_artifact_sha256",
                "model_training_summary_sha256",
                "request_model_artifact",
            }
            if (
                action not in EXACT_TOUCH_TEMPLATE_ACTIONS
                or donor.get("role") != "exact_touch_template_generator"
                or donor.get("generation_mode")
                != "exact_touch_template_generator_v1"
                or donor.get("generator_schema_version")
                != EXACT_TOUCH_TEMPLATE_SCHEMA_VERSION
                # The smoke reference route hands an event its own trajectory
                # back, so the template names the carrier.  The five-shot route
                # hands it one of the victim's five frozen events, which is by
                # construction a different event -- requiring equality there
                # would only be re-asserting that nothing was transformed.
                or (
                    donor.get("source_template_event_id") == str(row["event_id"])
                    if donor.get("request_source")
                    in FIVESHOT_MATERIAL_REQUEST_SOURCES
                    else donor.get("source_template_event_id")
                    != str(row["event_id"])
                )
                or (
                    donor.get("request_source")
                    in FIVESHOT_MATERIAL_REQUEST_SOURCES
                    and (
                        donor.get("source_material_cluster_id") is None
                        or not isinstance(
                            donor.get("source_material_shot_ordinal"), int
                        )
                    )
                )
                or len(str(donor.get("source_template_sha256", ""))) != 64
                or donor.get("human_replay") is not False
                or donor.get("runtime_donor_used") is not True
                or donor.get("model_used") is not False
                or donor.get("coordinate_clipping_used") is not False
                or donor.get("linear_touch_interpolation_used") is not False
                or donor.get("request_source")
                not in EXACT_TOUCH_REQUEST_SOURCES
                or donor.get("request_plan") is not None
                # A five-shot donor is a different real event of the same user,
                # so the output is no longer a byte copy of its template.  The
                # transform mode still has to be one the generator declares,
                # and the endpoint checks below remain exact.
                or donor.get("transform_mode")
                not in EXACT_TOUCH_TRANSFORM_MODES
                or not isinstance(donor.get("identity_transform"), bool)
                or (
                    donor.get("identity_transform")
                    is not (donor.get("transform_mode") == "identity_template")
                )
                or requested.shape != (2, 2)
                or raw_output.shape != (2, 2)
                or detector_output.shape != (2, 2)
                or not np.isfinite(requested).all()
                or not np.isfinite(raw_output).all()
                or not np.isfinite(detector_output).all()
                or not np.isfinite(declared_errors).all()
                or np.any(declared_errors < 0.0)
                or not np.allclose(
                    declared_errors, recomputed_errors, rtol=0.0, atol=1.0e-8
                )
                or not math.isclose(
                    float(donor.get("maximum_endpoint_error_px", np.nan)),
                    maximum_error,
                    rel_tol=0.0,
                    abs_tol=1.0e-8,
                )
                or maximum_error > 5.0e-4
                or requested_rows < 2
                or requested_rows != generated_rows
                or requested_duration <= 0.0
                or not math.isclose(
                    requested_duration,
                    generated_duration,
                    rel_tol=0.0,
                    abs_tol=1.0e-3,
                )
                or not 0.0 <= residual_scale <= 1.0
                or not isinstance(donor.get("tap_stationary_branch"), bool)
                or donor.get("conditioning_action") != action
                or not isinstance(
                    donor.get("conditioning_orientation_id"), int
                )
                or len(generator_digest) != 64
                or forbidden_model_fields & set(donor)
                or not _exact_touch_imu_provenance_is_valid(row)
            ):
                raise ReplayDatasetBuildError(
                    "exact touch template provenance is invalid"
                )
            if donor.get("request_source") == (
                "frozen_fiveshot_material_donor_drift_endpoints"
            ):
                try:
                    donor_drift = _provenance_finite_number(
                        donor["tap_donor_drift_px"],
                        name="five-shot tap donor drift",
                    )
                    requested_drift = _provenance_finite_number(
                        donor["tap_requested_drift_px"],
                        name="five-shot tap requested drift",
                    )
                    drift_scale = _provenance_finite_number(
                        donor["tap_donor_drift_scale"],
                        name="five-shot tap drift scale",
                    )
                    drift_limit = _provenance_finite_number(
                        donor["tap_drift_limit_px"],
                        name="five-shot tap drift limit",
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise ReplayDatasetBuildError(
                        "five-shot tap drift provenance is incomplete"
                    ) from exc
                # The request is the donor's own chord, so the requested drift
                # has to be both the recorded endpoint separation and, whenever
                # nothing was shrunk, the donor's measured drift.
                requested_separation = float(
                    np.linalg.norm(requested[1] - requested[0])
                )
                if (
                    action != "tap"
                    or donor_drift < 0.0
                    or requested_drift < 0.0
                    or not 0.0 <= drift_scale <= 1.0
                    or drift_limit != FIVESHOT_TAP_DRIFT_LIMIT_PX
                    or requested_drift > drift_limit
                    or not math.isclose(
                        requested_drift,
                        requested_separation,
                        rel_tol=0.0,
                        abs_tol=1.0e-6,
                    )
                    or (
                        drift_scale == 1.0
                        and not math.isclose(
                            requested_drift,
                            donor_drift,
                            rel_tol=0.0,
                            abs_tol=1.0e-6,
                        )
                    )
                ):
                    raise ReplayDatasetBuildError(
                        "five-shot tap drift provenance is invalid"
                    )
            observed_digest = exact_touch_generator_digest_by_path.get(
                generator_path
            )
            if observed_digest is None:
                if not generator_path.is_file():
                    raise ReplayDatasetBuildError(
                        "exact touch generator source disappeared"
                    )
                observed_digest = sha256_file(generator_path)
                exact_touch_generator_digest_by_path[generator_path] = (
                    observed_digest
                )
            if observed_digest != generator_digest:
                raise ReplayDatasetBuildError(
                    "exact touch generator source digest changed"
                )
            exact_touch_generated_events += 1
        elif method == FIVESHOT_KEYSTROKE_TOUCH_METHOD:
            action = str(row.get("action", ""))
            try:
                anchors = np.asarray(
                    donor["target_down_anchors_px"], dtype=np.float64
                )
                keycodes = [int(value) for value in donor["target_keycodes"]]
                holds = np.asarray(donor["hold_ms"], dtype=np.int64)
                flights = np.asarray(donor["flight_ms"], dtype=np.int64)
                segments = int(donor["contact_segments"])
                bounds = donor["timing_bounds_ms"]
                hold_bound = [int(value) for value in bounds["hold"]]
                flight_bound = [int(value) for value in bounds["flight"]]
                generator_path = Path(str(donor["generator_source"])).resolve()
                generator_digest = str(donor["generator_source_sha256"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ReplayDatasetBuildError(
                    "five-shot keystroke provenance is incomplete"
                ) from exc
            if (
                action != "keystroke"
                or donor.get("role") != "fiveshot_keystroke_rhythm_touch"
                or donor.get("generation_mode")
                != FIVESHOT_KEYSTROKE_GENERATION_MODE
                or donor.get("human_replay") is not False
                or donor.get("runtime_donor_used") is not True
                or donor.get("model_used") is not False
                or donor.get("coordinate_clipping_used") is not False
                or donor.get("linear_touch_interpolation_used") is not False
                # A keystroke contact carries no shape on the detector grid, so
                # requiring the source contact to be of the same key would be a
                # claim about information the observation does not hold.  The
                # flag is recorded so a reader can see the choice was made.
                or donor.get("keycode_matched_source_contact_required") is not False
                or donor.get("material_user_id") != str(row["user_id"])
                or not donor.get("material_source_event_ids")
                or int(donor.get("material_hold_count", 0)) < 1
                or anchors.shape != (len(keycodes), 2)
                or not np.isfinite(anchors).all()
                or len(holds) != len(keycodes)
                or len(flights) != max(0, len(keycodes) - 1)
                or hold_bound != [
                    ROBUST_KEYSTROKE_BOUNDS.hold_min_ms,
                    ROBUST_KEYSTROKE_BOUNDS.hold_max_ms,
                ]
                or flight_bound != [
                    ROBUST_KEYSTROKE_BOUNDS.flight_min_ms,
                    ROBUST_KEYSTROKE_BOUNDS.flight_max_ms,
                ]
                or np.any(holds < hold_bound[0])
                or np.any(holds > hold_bound[1])
                or (len(flights) and np.any(flights < flight_bound[0]))
                or (len(flights) and np.any(flights > flight_bound[1]))
                # A key can never produce more than one contact, and the event
                # has to leave at least one.  Fewer segments than keys is not a
                # defect: two presses less than two detector periods apart merge
                # into one contact for a genuine event exactly as they do here,
                # and the merge is only admissible because it is counted.
                or segments > len(keycodes)
                or segments < 1
                or int(donor.get("merged_contact_segments", -1))
                != len(keycodes) - segments
                or int(donor.get("contact_samples", 0)) < segments
                or forbidden_model_fields & set(donor)
            ):
                raise ReplayDatasetBuildError(
                    "five-shot keystroke provenance is invalid"
                )
            observed_digest = exact_touch_generator_digest_by_path.get(
                generator_path
            )
            if observed_digest is None:
                if not generator_path.is_file():
                    raise ReplayDatasetBuildError(
                        "five-shot keystroke generator source disappeared"
                    )
                observed_digest = sha256_file(generator_path)
                exact_touch_generator_digest_by_path[generator_path] = (
                    observed_digest
                )
            if observed_digest != generator_digest:
                raise ReplayDatasetBuildError(
                    "five-shot keystroke generator source digest changed"
                )
            fiveshot_keystroke_events += 1
        elif method == PINCH_AREA_SIMILARITY_REBUILD_METHOD:
            action = str(row.get("action", ""))
            try:
                requested_points = np.asarray(
                    donor["requested_area_points_px"], dtype=np.float64
                )
                delivered_points = np.asarray(
                    donor["delivered_area_points_px"], dtype=np.float64
                )
                declared_extent_error = _provenance_finite_number(
                    donor["delivered_area_extent_error_px"],
                    name="pinch area extent error",
                )
                requested_span = _provenance_finite_number(
                    donor["requested_area_span_px"], name="pinch requested span"
                )
                source_span = _provenance_finite_number(
                    donor["source_widest_span_px"], name="pinch source span"
                )
                similarity_scale = _provenance_finite_number(
                    donor["similarity_scale"], name="pinch similarity scale"
                )
                similarity_rotation = _provenance_finite_number(
                    donor["similarity_rotation_rad"],
                    name="pinch similarity rotation",
                )
                requested_axis = _provenance_finite_number(
                    donor["requested_area_axis_rad"], name="pinch requested axis"
                )
                source_axis = _provenance_finite_number(
                    donor["source_widest_axis_rad"], name="pinch source axis"
                )
                source_percent = _provenance_finite_number(
                    donor["source_percent"], name="pinch source percent"
                )
                requested_duration = _provenance_finite_number(
                    donor["requested_raw_duration_ms"],
                    name="pinch requested duration",
                )
                generated_duration = _provenance_finite_number(
                    donor["generated_raw_duration_ms"],
                    name="pinch generated duration",
                )
                generator_path = Path(str(donor["generator_source"])).resolve()
                generator_digest = str(donor["generator_source_sha256"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ReplayDatasetBuildError(
                    "pinch area similarity provenance is incomplete"
                ) from exc
            recomputed_extent_error = (
                float(
                    np.max(
                        np.linalg.norm(delivered_points - requested_points, axis=1)
                    )
                )
                if requested_points.shape == (2, 2)
                and delivered_points.shape == (2, 2)
                else float("inf")
            )
            if (
                action not in PINCH_AREA_SIMILARITY_ACTIONS
                or donor.get("role") != "fiveshot_pinch_area_similarity"
                or donor.get("generation_mode")
                != PINCH_AREA_SIMILARITY_GENERATION_MODE
                or donor.get("source_template_event_id") == str(row["event_id"])
                or len(str(donor.get("source_template_sha256", ""))) != 64
                or donor.get("human_replay") is not False
                or donor.get("runtime_donor_used") is not True
                or donor.get("model_used") is not False
                or donor.get("coordinate_clipping_used") is not False
                or donor.get("linear_touch_interpolation_used") is not False
                or donor.get("differential_deformation_used") is not False
                or donor.get("off_screen_sample_count") != 0
                or donor.get("request_source") not in PINCH_AREA_REQUEST_SOURCES
                or donor.get("request_plan") is not None
                or donor.get("source_scale_direction") not in {"in", "out"}
                or donor.get("pinch_pointer_order") not in PINCH_POINTER_ORDERS
                or not isinstance(donor.get("material_substituted"), bool)
                or not isinstance(
                    donor.get("assigned_material_shot_ordinal"), int
                )
                or (
                    donor.get("material_substituted")
                    is not (
                        donor.get("assigned_material_shot_ordinal")
                        != donor.get("source_material_shot_ordinal")
                    )
                )
                or requested_points.shape != (2, 2)
                or delivered_points.shape != (2, 2)
                or donor.get("requested_area_moment") not in {"start", "end"}
                or donor.get("source_widest_moment") not in {"start", "end"}
                or donor.get("off_screen_finger_endpoint_count") != 0
                or requested_span <= 0.0
                or source_span <= 0.0
                or source_percent <= 0.0
                # The similarity is fully determined by the request and the
                # recording, so both parameters have to reproduce exactly.  A
                # scale gate is deliberately absent: five recordings cannot
                # cover every requested area size, and refusing the tail would
                # silently drop events instead of reporting the rescale.
                or not math.isclose(
                    similarity_scale,
                    requested_span / source_span,
                    rel_tol=1.0e-12,
                    abs_tol=0.0,
                )
                or not math.isclose(
                    similarity_rotation,
                    _wrap_pinch_angle(requested_axis - source_axis),
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                )
                or declared_extent_error < 0.0
                or not math.isclose(
                    declared_extent_error,
                    recomputed_extent_error,
                    rel_tol=0.0,
                    abs_tol=1.0e-8,
                )
                or recomputed_extent_error > 5.0e-4
                or requested_duration <= 0.0
                or not math.isclose(
                    requested_duration,
                    generated_duration,
                    rel_tol=0.0,
                    abs_tol=1.0e-3,
                )
                or donor.get("requested_raw_row_count")
                != donor.get("generated_raw_row_count")
                or forbidden_model_fields & set(donor)
                or row.get("input_imu_sha256") != row.get("output_imu_sha256")
            ):
                raise ReplayDatasetBuildError(
                    "pinch area similarity provenance is invalid"
                )
            observed_digest = exact_touch_generator_digest_by_path.get(
                generator_path
            )
            if observed_digest is None:
                if not generator_path.is_file():
                    raise ReplayDatasetBuildError(
                        "pinch area similarity source disappeared"
                    )
                observed_digest = sha256_file(generator_path)
                exact_touch_generator_digest_by_path[generator_path] = (
                    observed_digest
                )
            if observed_digest != generator_digest:
                raise ReplayDatasetBuildError(
                    "pinch area similarity source digest changed"
                )
            pinch_area_similarity_events += 1
        elif method == CONDITIONAL_TOUCH_REBUILD_METHOD:
            action = str(row.get("action", ""))
            try:
                requested = np.asarray(
                    (
                        donor["requested_start_px"],
                        donor["requested_end_px"],
                    ),
                    dtype=np.float64,
                )
                raw_output = np.asarray(
                    (
                        donor["raw_output_start_px"],
                        donor["raw_output_end_px"],
                    ),
                    dtype=np.float64,
                )
                detector_output = np.asarray(
                    (
                        donor["detector_output_start_px"],
                        donor["detector_output_end_px"],
                    ),
                    dtype=np.float64,
                )
                declared_errors = np.asarray(
                    (
                        donor["raw_start_error_px"],
                        donor["raw_end_error_px"],
                        donor["detector_start_error_px"],
                        donor["detector_end_error_px"],
                    ),
                    dtype=np.float64,
                )
                generator_seed = _provenance_native_int(
                    donor["generator_seed"], name="conditional touch seed"
                )
                requested_rows = _provenance_native_int(
                    donor["requested_raw_row_count"],
                    name="conditional touch requested raw rows",
                )
                generated_rows = _provenance_native_int(
                    donor["generated_raw_row_count"],
                    name="conditional touch generated raw rows",
                )
                requested_duration = _provenance_finite_number(
                    donor["requested_raw_duration_ms"],
                    name="conditional touch requested raw duration",
                )
                generated_duration = _provenance_finite_number(
                    donor["generated_raw_duration_ms"],
                    name="conditional touch generated raw duration",
                )
                residual_scale = _provenance_finite_number(
                    donor["residual_scale"],
                    name="conditional touch residual scale",
                )
                model_path = Path(str(donor["model_artifact"])).resolve()
                model_file_digest = str(donor["model_file_sha256"])
                model_canonical_digest = str(
                    donor["model_canonical_artifact_sha256"]
                )
                summary_digest = str(
                    donor["model_training_summary_sha256"]
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ReplayDatasetBuildError(
                    "conditional touch generator provenance is incomplete"
                ) from exc
            raw_errors = np.linalg.norm(raw_output - requested, axis=1)
            detector_errors = np.linalg.norm(
                detector_output - requested, axis=1
            )
            recomputed_errors = np.concatenate((raw_errors, detector_errors))
            maximum_error = float(np.max(recomputed_errors))
            generation_mode = donor.get("generation_mode")
            if generation_mode is None and donor.get("role") == (
                "frozen_conditional_touch_generator_model"
            ):
                generation_mode = "frozen_conditional_touch_generator"
            template_transport = generation_mode == (
                "frozen_smoke_reference_template_transport"
            )
            model_generation = generation_mode == (
                "frozen_conditional_touch_generator"
            )
            provenance_role_valid = bool(
                (
                    template_transport
                    and action in {"tap", "swipe"}
                    and donor.get("role")
                    == "frozen_smoke_reference_touch_template"
                    and donor.get("runtime_donor_used") is True
                    and donor.get("model_used") is False
                    and donor.get("source_template_event_id")
                    == str(row["event_id"])
                )
                or (
                    model_generation
                    and donor.get("role")
                    == "frozen_conditional_touch_generator_model"
                    and donor.get("runtime_donor_used") is False
                    and donor.get("model_used", True) is True
                    and donor.get("source_template_event_id") is None
                )
            )
            forbidden_donor_fields = {
                "primitive_id",
                "source_event_id",
                "source_user_id",
                "source_session_id",
                "raw_archive",
            }
            if (
                action not in CONDITIONAL_TOUCH_ACTIONS
                or not provenance_role_valid
                or donor.get("human_replay") is not False
                or donor.get("coordinate_clipping_used") is not False
                or donor.get("linear_touch_interpolation_used") is not False
                or not isinstance(donor.get("model_schema_version"), str)
                or not donor.get("model_schema_version")
                or requested.shape != (2, 2)
                or raw_output.shape != (2, 2)
                or detector_output.shape != (2, 2)
                or not np.isfinite(requested).all()
                or not np.isfinite(raw_output).all()
                or not np.isfinite(detector_output).all()
                or not np.isfinite(declared_errors).all()
                or np.any(declared_errors < 0.0)
                or not np.allclose(
                    declared_errors,
                    recomputed_errors,
                    rtol=0.0,
                    atol=1.0e-8,
                )
                or not math.isclose(
                    float(donor.get("maximum_endpoint_error_px", np.nan)),
                    maximum_error,
                    rel_tol=0.0,
                    abs_tol=1.0e-8,
                )
                or maximum_error > 5.0e-4
                or generator_seed < 0
                or requested_rows < 2
                or requested_rows != generated_rows
                or requested_duration <= 0.0
                or not math.isclose(
                    requested_duration,
                    generated_duration,
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                )
                or residual_scale < 0.0
                or not isinstance(donor.get("tap_stationary_branch"), bool)
                or donor.get("conditioning_action") != action
                or not isinstance(
                    donor.get("conditioning_orientation_id"), int
                )
                or len(model_file_digest) != 64
                or len(model_canonical_digest) != 64
                or len(summary_digest) != 64
                or forbidden_donor_fields & set(donor)
                or row.get("input_imu_sha256")
                != row.get("output_imu_sha256")
            ):
                raise ReplayDatasetBuildError(
                    "conditional touch generator provenance is invalid"
                )
            direction = donor.get("conditioning_direction")
            if action != "tap" and (
                not isinstance(direction, str)
                or not direction
                or donor.get("realized_direction") != direction
            ):
                raise ReplayDatasetBuildError(
                    "conditional gesture provenance changed its direction"
                )
            observed_digest = conditional_touch_model_digest_by_path.get(
                model_path
            )
            if observed_digest is None:
                if not model_path.is_file():
                    raise ReplayDatasetBuildError(
                        "conditional touch model artifact disappeared"
                    )
                observed_digest = sha256_file(model_path)
                conditional_touch_model_digest_by_path[model_path] = (
                    observed_digest
                )
            if observed_digest != model_file_digest:
                raise ReplayDatasetBuildError(
                    "conditional touch model provenance digest changed"
                )
            request_plan_value = donor.get("request_plan")
            request_plan = None
            if request_plan_value is not None:
                try:
                    request_plan = ConditionalTouchRequestPlan.from_json_dict(
                        request_plan_value
                    )
                    request_model_path = Path(
                        str(donor["request_model_artifact"])
                    ).resolve()
                except (KeyError, TypeError, ValueError) as exc:
                    raise ReplayDatasetBuildError(
                        "conditional touch request provenance is invalid"
                    ) from exc
                if (
                    request_plan.carrier_event_id != str(row["event_id"])
                    or request_plan.action != action
                    or request_plan.orientation_id
                    != int(donor["conditioning_orientation_id"])
                    or not np.array_equal(
                        np.asarray(request_plan.sampled_start_xy_px),
                        requested[0],
                    )
                    or not np.array_equal(
                        np.asarray(request_plan.sampled_end_xy_px),
                        requested[1],
                    )
                    or request_plan.sampled_direction != direction
                    or request_plan.request_model_schema_version
                    == ""
                    or request_plan.request_model_source_fingerprint_sha256
                    != CONDITIONAL_TOUCH_REQUEST_SOURCE_FINGERPRINT_SHA256
                    or request_plan.request_model_artifact_sha256
                    != request_plan.request_model_file_sha256
                    or not request_model_path.is_file()
                    or sha256_file(request_model_path)
                    != request_plan.request_model_file_sha256
                ):
                    raise ReplayDatasetBuildError(
                        "conditional touch request plan changed"
                    )
            if action == "tap":
                if request_plan is None:
                    if donor.get("request_source") == (
                        "frozen_smoke_reference_exact_endpoints"
                    ):
                        expected_realized = (
                            "stationary" if direction is None else direction
                        )
                        if donor.get("realized_direction") != expected_realized:
                            raise ReplayDatasetBuildError(
                                "frozen tap request changed its direction"
                            )
                    elif direction is not None or not np.allclose(
                        requested[0], requested[1], rtol=0.0, atol=1.0e-12
                    ):
                        raise ReplayDatasetBuildError(
                            "conditional tap provenance changed its input coordinate"
                        )
                elif donor.get("realized_direction") != (
                    "stationary" if direction is None else direction
                ):
                    raise ReplayDatasetBuildError(
                        "conditional tap request direction changed"
                    )
            conditional_touch_generated_events += 1
        elif method in {
            "train_raw_action_replay",
            "train_raw_action_isometric_replay",
            "train_raw_action_pinch_endpoint_replay",
        }:
            primitive = str(donor["primitive_id"])
            if primitive in primitive_by_split[split]:
                raise ReplayDatasetBuildError("action replay donor was reused")
            primitive_by_split[split].add(primitive)
            all_donor_events.add(str(donor["source_event_id"]))
            if method == "train_raw_action_isometric_replay":
                try:
                    matrix = np.asarray(
                        donor.get("spatial_matrix_xy"), dtype=np.float64
                    )
                    translation = np.asarray(
                        donor.get("translation_px"), dtype=np.float64
                    )
                    requested_anchor = np.asarray(
                        donor.get("requested_anchor_px"), dtype=np.float64
                    )
                    output_anchor = np.asarray(
                        donor.get("output_anchor_px"), dtype=np.float64
                    )
                except (TypeError, ValueError) as exc:
                    raise ReplayDatasetBuildError(
                        "isometric action replay provenance is incomplete"
                    ) from exc
                gram = matrix.T @ matrix if matrix.shape == (2, 2) else matrix
                scale_sq = (
                    float(np.trace(gram) / 2.0) if matrix.shape == (2, 2) else np.nan
                )
                if (
                    matrix.shape != (2, 2)
                    or not np.isfinite(matrix).all()
                    or not np.isfinite(scale_sq)
                    or scale_sq <= 0.0
                    or not np.allclose(gram, np.eye(2) * scale_sq, atol=1.0e-7)
                    or translation.shape != (2,)
                    or requested_anchor.shape != (2,)
                    or output_anchor.shape != (2,)
                    or not np.isfinite(translation).all()
                    or not np.isfinite(requested_anchor).all()
                    or not np.isfinite(output_anchor).all()
                    or not donor.get("spatial_transform_name")
                    or float(donor.get("time_warp_ratio", np.nan)) != 1.0
                    or donor.get("scale_used") is not (
                        not math.isclose(math.sqrt(scale_sq), 1.0, abs_tol=1.0e-12)
                    )
                    or donor.get("coordinate_clipping_used") is not False
                    or not np.isfinite(
                        float(donor.get("anchor_error_px", np.nan))
                    )
                    or not np.isfinite(
                        float(donor.get("spatial_scale", math.sqrt(scale_sq)))
                    )
                    or not np.isfinite(float(donor.get("endpoint_error_px", 0.0)))
                ):
                    raise ReplayDatasetBuildError(
                        "isometric action replay provenance is incomplete"
                    )
            elif method == "train_raw_action_pinch_endpoint_replay":
                try:
                    source_start = np.asarray(
                        donor["pinch_source_start_points_px"],
                        dtype=np.float64,
                    )
                    source_end = np.asarray(
                        donor["pinch_source_end_points_px"],
                        dtype=np.float64,
                    )
                    requested_start = np.asarray(
                        donor["pinch_requested_start_points_px"],
                        dtype=np.float64,
                    )
                    requested_end = np.asarray(
                        donor["pinch_requested_end_points_px"],
                        dtype=np.float64,
                    )
                    scales = np.asarray(
                        (
                            donor["pinch_center_scale"],
                            donor["pinch_start_span_scale"],
                            donor["pinch_end_span_scale"],
                        ),
                        dtype=np.float64,
                    )
                    errors = np.asarray(
                        (
                            donor["pinch_start_endpoint_error_px"],
                            donor["pinch_end_endpoint_error_px"],
                            donor["pinch_maximum_endpoint_error_px"],
                        ),
                        dtype=np.float64,
                    )
                    requested_anchor = np.asarray(
                        donor["requested_anchor_px"], dtype=np.float64
                    )
                    requested_endpoint = np.asarray(
                        donor["requested_endpoint_px"], dtype=np.float64
                    )
                    translation = np.asarray(
                        donor["translation_px"], dtype=np.float64
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise ReplayDatasetBuildError(
                        "pinch endpoint replay provenance is incomplete"
                    ) from exc
                if any(
                    value.shape != (2, 2)
                    for value in (
                        source_start,
                        source_end,
                        requested_start,
                        requested_end,
                    )
                ):
                    raise ReplayDatasetBuildError(
                        "pinch endpoint replay provenance is incomplete"
                    )
                expected_span_scales = np.asarray(
                    (
                        np.linalg.norm(requested_start[1] - requested_start[0])
                        / np.linalg.norm(source_start[1] - source_start[0]),
                        np.linalg.norm(requested_end[1] - requested_end[0])
                        / np.linalg.norm(source_end[1] - source_end[0]),
                    ),
                    dtype=np.float64,
                )
                expected_deformation = float(
                    np.max(np.abs(np.log(scales)))
                )
                if (
                    donor.get("role")
                    != "train_raw_action_pinch_endpoint_replay"
                    or not np.isfinite(source_start).all()
                    or not np.isfinite(source_end).all()
                    or not np.isfinite(requested_start).all()
                    or not np.isfinite(requested_end).all()
                    or not np.isfinite(scales).all()
                    or np.any(scales < 0.80 - 1.0e-12)
                    or np.any(scales > 1.25 + 1.0e-12)
                    or not np.allclose(
                        scales[1:], expected_span_scales, atol=1.0e-9
                    )
                    or not np.isfinite(errors).all()
                    or np.any(errors < 0.0)
                    or np.any(errors > 1.0e-6)
                    or not math.isclose(
                        float(donor["pinch_deformation_score"]),
                        expected_deformation,
                        abs_tol=1.0e-12,
                    )
                    or float(donor.get("pinch_minimum_scale", np.nan))
                    != 0.80
                    or float(donor.get("pinch_maximum_scale", np.nan))
                    != 1.25
                    or donor.get("spatial_transform_name")
                    != "pinch_bounded_endpoint_residual"
                    or donor.get("spatial_matrix_xy") is not None
                    or donor.get("coordinate_clipping_used") is not False
                    or donor.get("pixel_lattice_correction_used") is not False
                    or float(donor.get("time_warp_ratio", np.nan)) != 1.0
                    or requested_anchor.shape != (2,)
                    or requested_endpoint.shape != (2,)
                    or translation.shape != (2,)
                    or not np.isfinite(translation).all()
                    or not np.allclose(
                        requested_anchor,
                        np.mean(requested_start, axis=0),
                        atol=1.0e-9,
                    )
                    or not np.allclose(
                        requested_endpoint,
                        np.mean(requested_end, axis=0),
                        atol=1.0e-9,
                    )
                    or donor.get("scale_used") is not any(
                        not math.isclose(
                            float(value), 1.0, abs_tol=1.0e-12
                        )
                        for value in scales
                    )
                ):
                    raise ReplayDatasetBuildError(
                        "pinch endpoint replay provenance is incomplete"
                    )
        elif method == "train_raw_tap_replay":
            source_event = str(donor["source_event_id"])
            if source_event in tap_by_split[split]:
                raise ReplayDatasetBuildError("tap replay donor was reused")
            tap_by_split[split].add(source_event)
            all_donor_events.add(source_event)
        elif method == "bound_fake_tap_android_zoh":
            if donor.get("human_replay") is not False:
                raise ReplayDatasetBuildError(
                    "bound fake tap provenance misstates human replay"
                )
        else:
            raise ReplayDatasetBuildError(f"unknown fake provenance method {method}")
    for groups in (
        primitive_by_split,
        tap_by_split,
    ):
        for index, left in enumerate(SPLITS):
            for right in SPLITS[index + 1 :]:
                if groups[left] & groups[right]:
                    raise ReplayDatasetBuildError(
                        f"donor family leakage between {left}/{right}"
                    )
    overlap = all_donor_events & selected_genuine_source_event_ids
    if overlap:
        raise ReplayDatasetBuildError(
            f"replay donors overlap selected genuine sources: {len(overlap)}"
        )
    return {
        "exact_touch_template_generated_events": exact_touch_generated_events,
        "pinch_area_similarity_generated_events": pinch_area_similarity_events,
        "fiveshot_keystroke_touch_events": fiveshot_keystroke_events,
        "exact_touch_template_generator_files": len(
            exact_touch_generator_digest_by_path
        ),
        "conditional_touch_generated_events": (
            conditional_touch_generated_events
        ),
        "conditional_touch_model_files": len(
            conditional_touch_model_digest_by_path
        ),
        "action_primitives": sum(len(value) for value in primitive_by_split.values()),
        "tap_primitives": sum(len(value) for value in tap_by_split.values()),
        "donor_overlap_across_output_splits": 0,
        "donor_overlap_selected_genuine": 0,
    }


def _joint_events_root(release: Mapping[str, Any]) -> Path:
    manifest = Path(str(release.get("joint_fake_manifest", ""))).resolve()
    root = manifest.parent / "events"
    if not root.is_dir():
        raise ReplayDatasetBuildError(f"joint fake event directory is missing: {root}")
    return root


def _split_train_users(split_path: Path) -> tuple[int, ...]:
    value = _read_json(split_path)
    users = value.get("train_users")
    if not isinstance(users, list) or not users:
        raise ReplayDatasetBuildError("split file has no train users")
    return tuple(int(user) for user in users)


def _candidate_users(
    shards: Sequence[InputShard], *, users_per_split: int
) -> list[InputShard]:
    selected: list[InputShard] = []
    for source in shards:
        shard = _load_shard(source, signals=False)
        labels = np.asarray(shard.arrays["label"], dtype=np.int64)
        actions = np.asarray(shard.arrays["action"]).astype(str)
        if all(
            np.any((labels == label) & (actions == action))
            for action in ACTIONS
            for label in (0, 1)
        ):
            selected.append(source)
        if len(selected) == users_per_split:
            return selected
    raise ReplayDatasetBuildError(
        f"split has fewer than {users_per_split} complete smoke users"
    )


SmokeReferenceKey = tuple[str, str, str, int]
SmokeTouchRequest = tuple[
    tuple[float, float], tuple[float, float], str | None
]


def _smoke_reference_selection_sha256(
    selection: Mapping[SmokeReferenceKey, Sequence[str]],
) -> str:
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        split_keys = sorted(
            (key for key in selection if key[0] == split),
            key=lambda key: (
                key[1], ACTIONS.index(key[2]), int(key[3])
            ),
        )
        for key in split_keys:
            _, user_id, action, label = key
            rows.append(
                {
                    "split": split,
                    "user_id": user_id,
                    "action": action,
                    "label": int(label),
                    "event_ids": [str(value) for value in selection[key]],
                }
            )
    encoded = json.dumps(
        rows, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_smoke_reference_selection(
    *,
    reference_manifest: str | Path,
    chosen_sources: Mapping[str, Sequence[InputShard]],
    events_per_label_action_user: int,
) -> tuple[
    dict[SmokeReferenceKey, tuple[str, ...]],
    dict[str, Any],
    dict[str, SmokeTouchRequest],
    dict[str, np.ndarray],
]:
    """Freeze exact smoke event IDs from a prior sharded detector manifest."""

    manifest = Path(reference_manifest).resolve()
    if not manifest.is_file():
        raise ReplayDatasetBuildError(
            f"smoke reference manifest is missing: {manifest}"
        )
    try:
        rows = _load_manifest(manifest)
    except (EventPadError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise ReplayDatasetBuildError(
            f"smoke reference manifest is invalid: {exc}"
        ) from exc
    if any(
        row.get("schema_version") != INPUT_MANIFEST_SCHEMA
        for row in rows.values()
    ):
        raise ReplayDatasetBuildError(
            "smoke reference must use the sharded detector manifest schema"
        )
    reference_provenance_path = manifest.parent / "provenance.jsonl"
    reference_orientation_by_event: dict[str, int] = {}
    if not reference_provenance_path.is_file():
        raise ReplayDatasetBuildError(
            "smoke reference provenance is missing"
        )
    with reference_provenance_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            donor = row.get("donor", {})
            target_binding = (
                donor.get("target_binding", {})
                if isinstance(donor, Mapping)
                else {}
            )
            orientation = target_binding.get(
                "orientation_id",
                donor.get("orientation_id")
                if isinstance(donor, Mapping)
                else None,
            )
            if orientation is not None:
                reference_orientation_by_event[str(row["event_id"])] = int(
                    orientation
                )

    quota = int(events_per_label_action_user)
    selected_users = {
        split: tuple(source.user_id for source in chosen_sources[split])
        for split in SPLITS
    }
    expected_groups = sum(len(values) for values in selected_users.values()) * (
        len(ACTIONS) * 2
    )
    expected_events = expected_groups * quota
    for split in SPLITS:
        reference_users = tuple(str(value) for value in rows[split]["user_ids"])
        if reference_users != selected_users[split]:
            raise ReplayDatasetBuildError(
                "smoke reference users must exactly equal the selected smoke "
                f"users in {split}"
            )
    if sum(int(rows[split]["events"]) for split in SPLITS) != expected_events:
        raise ReplayDatasetBuildError(
            "smoke reference manifest event count must exactly equal the "
            "requested split/user/action/label quotas"
        )
    reference_by_split_user: dict[tuple[str, str], InputShard] = {}
    for split in SPLITS:
        for shard in rows[split]["shards"]:
            user_id = str(shard["user_id"])
            key = (split, user_id)
            if key in reference_by_split_user:
                raise ReplayDatasetBuildError(
                    "smoke reference repeats a split/user shard"
                )
            reference_by_split_user[key] = InputShard(
                split=split,
                user_id=user_id,
                path=Path(shard["source"]).resolve(),
                sha256=str(shard["source_sha256"]),
                manifest_row=dict(shard),
            )

    current_binding_by_event: dict[str, tuple[str, str, str, int]] = {}
    for split in SPLITS:
        for source in chosen_sources[split]:
            shard = _load_shard(source, signals=False)
            event_ids = np.asarray(shard.arrays["event_id"]).astype(str)
            actions = np.asarray(shard.arrays["action"]).astype(str)
            labels = np.asarray(shard.arrays["label"], dtype=np.int64)
            users = np.asarray(shard.arrays["user_id"]).astype(str)
            if (
                not (
                    len(event_ids) == len(actions) == len(labels) == len(users)
                )
                or np.any(users != source.user_id)
            ):
                raise ReplayDatasetBuildError(
                    "current smoke source metadata is malformed"
                )
            for event_id, action, label in zip(event_ids, actions, labels):
                identity = str(event_id)
                binding = (
                    split,
                    source.user_id,
                    str(action),
                    int(label),
                )
                if identity in current_binding_by_event:
                    raise ReplayDatasetBuildError(
                        "current smoke source repeats an event ID"
                    )
                current_binding_by_event[identity] = binding

    selection: dict[SmokeReferenceKey, tuple[str, ...]] = {}
    touch_requests: dict[str, SmokeTouchRequest] = {}
    touch_templates: dict[str, np.ndarray] = {}
    frozen_event_ids: set[str] = set()
    reference_candidate_events = 0
    for split in SPLITS:
        for user_id in selected_users[split]:
            reference_source = reference_by_split_user.get((split, user_id))
            if reference_source is None:
                raise ReplayDatasetBuildError(
                    f"smoke reference lacks selected user {split}/{user_id}"
                )
            shard = _load_shard(reference_source, signals=True)
            event_ids = np.asarray(shard.arrays["event_id"]).astype(str)
            actions = np.asarray(shard.arrays["action"]).astype(str)
            labels = np.asarray(shard.arrays["label"], dtype=np.int64)
            users = np.asarray(shard.arrays["user_id"]).astype(str)
            if (
                not (
                    len(event_ids) == len(actions) == len(labels) == len(users)
                )
                or np.any(users != user_id)
                or len(event_ids) != len(set(event_ids.tolist()))
            ):
                raise ReplayDatasetBuildError(
                    "smoke reference shard metadata is malformed"
                )
            reference_candidate_events += len(event_ids)
            for action in ACTIONS:
                for label in (0, 1):
                    candidate_indices = np.flatnonzero(
                        (actions == action) & (labels == label)
                    )
                    candidates = event_ids[candidate_indices].tolist()
                    if len(candidates) != quota:
                        raise ReplayDatasetBuildError(
                            "smoke reference group event count must exactly equal "
                            f"its quota: {split}/{user_id}/{action}/{label} has "
                            f"{len(candidates)}, expected {quota}"
                        )
                    frozen = tuple(str(value) for value in candidates)
                    key = (split, user_id, action, label)
                    for candidate_index, event_id in zip(
                        candidate_indices, frozen
                    ):
                        if event_id in frozen_event_ids:
                            raise ReplayDatasetBuildError(
                                "smoke reference freezes one event ID twice"
                            )
                        expected_binding = current_binding_by_event.get(event_id)
                        if expected_binding != key:
                            raise ReplayDatasetBuildError(
                                "smoke reference event is missing from the current "
                                "input or changed split/user/action/label binding"
                            )
                        frozen_event_ids.add(event_id)
                        if label == 1 and action in {"tap", "swipe"}:
                            _, trajectory = shard.event_signal(
                                int(candidate_index)
                            )
                            orientation_id = reference_orientation_by_event.get(
                                event_id
                            )
                            if orientation_id is None:
                                raise ReplayDatasetBuildError(
                                    "smoke reference touch orientation is missing"
                                )
                            dimensions = np.asarray(
                                screen_dimensions_for_orientation(
                                    orientation_id
                                ),
                                dtype=np.float64,
                            )
                            endpoints = (
                                np.asarray(
                                    trajectory[[0, -1], 1:3],
                                    dtype=np.float64,
                                )
                                * dimensions[None, :]
                            )
                            delta = endpoints[1] - endpoints[0]
                            if np.array_equal(endpoints[0], endpoints[1]):
                                request_direction = None
                            else:
                                labels8 = (
                                    "right", "down_right", "down", "down_left",
                                    "left", "up_left", "up", "up_right",
                                )
                                angle = float(np.arctan2(delta[1], delta[0]))
                                request_direction = labels8[
                                    int(
                                        np.floor(
                                            (angle + np.pi / 8.0)
                                            / (np.pi / 4.0)
                                        )
                                    )
                                    % len(labels8)
                                ]
                            touch_requests[event_id] = (
                                (
                                    float(endpoints[0, 0]),
                                    float(endpoints[0, 1]),
                                ),
                                (
                                    float(endpoints[1, 0]),
                                    float(endpoints[1, 1]),
                                ),
                                request_direction,
                            )
                            touch_templates[event_id] = np.asarray(
                                trajectory, dtype=np.float32
                            ).copy()
                    selection[key] = frozen

    if (
        len(selection) != expected_groups
        or len(frozen_event_ids) != expected_events
        or reference_candidate_events != expected_events
    ):
        raise ReplayDatasetBuildError(
            "smoke reference selection coverage is incomplete"
        )
    digest = _smoke_reference_selection_sha256(selection)
    return selection, {
        "status": "frozen_pending_rebuild",
        "reference_manifest": str(manifest),
        "reference_manifest_sha256": sha256_file(manifest),
        "users_by_split": {
            split: list(selected_users[split]) for split in SPLITS
        },
        "groups": len(selection),
        "quota_per_split_user_action_label": quota,
        "reference_candidate_events": reference_candidate_events,
        "frozen_events": len(frozen_event_ids),
        "frozen_event_ids_sha256": digest,
        "current_input_binding_exact_match": True,
        "frozen_touch_endpoint_requests": len(touch_requests),
    }, touch_requests, touch_templates


FULL_USER_COUNTS = {"train": 70, "development": 10, "test": 20}
FULL_FAKE_COUNTS = {"train": 70_000, "development": 10_000, "test": 20_000}
FULL_FAKE_ACTION_COUNTS = {
    "train": 14_000,
    "development": 2_000,
    "test": 4_000,
}


def _validate_full_input_selection(
    shards_by_split: Mapping[str, Sequence[InputShard]],
) -> dict[str, Any]:
    """Prove that selecting all input indices is the fixed full-100k set."""

    seen_users: set[str] = set()
    seen_events: set[str] = set()
    counts: dict[str, int] = {}
    for split in SPLITS:
        sources = list(shards_by_split[split])
        source_users = [source.user_id for source in sources]
        if (
            len(sources) != FULL_USER_COUNTS[split]
            or len(source_users) != len(set(source_users))
            or seen_users.intersection(source_users)
        ):
            raise ReplayDatasetBuildError(
                f"full input violates fixed disjoint user policy in {split}"
            )
        seen_users.update(source_users)
        split_fake = 0
        for source in sources:
            shard = _load_shard(source, signals=False)
            labels = np.asarray(shard.arrays["label"], dtype=np.int64)
            actions = np.asarray(shard.arrays["action"]).astype(str)
            users = np.asarray(shard.arrays["user_id"]).astype(str)
            event_ids = np.asarray(shard.arrays["event_id"]).astype(str)
            offsets = np.asarray(shard.arrays["offsets"], dtype=np.int64)
            if (
                len(labels) < 1
                or actions.shape != labels.shape
                or users.shape != labels.shape
                or event_ids.shape != labels.shape
                or offsets.shape != (len(labels) + 1,)
                or int(offsets[0]) != 0
                or np.any(np.diff(offsets) < 2)
                or np.any(~np.isin(labels, (0, 1)))
                or np.any(~np.isin(actions, ACTIONS))
                or np.any(users != source.user_id)
            ):
                raise ReplayDatasetBuildError(
                    f"full input shard contract changed: {source.path}"
                )
            duplicate = seen_events.intersection(event_ids.tolist())
            if duplicate or len(event_ids) != len(set(event_ids.tolist())):
                raise ReplayDatasetBuildError("full input event IDs are not unique")
            seen_events.update(event_ids.tolist())
            for action in ACTIONS:
                for label in (0, 1):
                    name = (
                        f"{split}/{action}/"
                        f"{'genuine' if label == 0 else 'fake'}"
                    )
                    value = int(np.sum((actions == action) & (labels == label)))
                    counts[name] = counts.get(name, 0) + value
            split_fake += int(np.sum(labels == 1))
        if split_fake != FULL_FAKE_COUNTS[split]:
            raise ReplayDatasetBuildError(
                f"full input has {split_fake}, not {FULL_FAKE_COUNTS[split]}, "
                f"fake events in {split}"
            )
        for action in ACTIONS:
            action_fake = counts[f"{split}/{action}/fake"]
            if action_fake != FULL_FAKE_ACTION_COUNTS[split]:
                raise ReplayDatasetBuildError(
                    f"full input {split}/{action} fake count is {action_fake}, "
                    f"not {FULL_FAKE_ACTION_COUNTS[split]}"
                )
    fake = sum(value for key, value in counts.items() if key.endswith("/fake"))
    genuine = sum(
        value for key, value in counts.items() if key.endswith("/genuine")
    )
    if fake != 100_000 or len(seen_users) != 100:
        raise ReplayDatasetBuildError("full input is not the fixed 100-user/100k set")
    return {
        "events": fake + genuine,
        "fake_events": fake,
        "genuine_events": genuine,
        "users": len(seen_users),
        "action_label_counts": dict(sorted(counts.items())),
        "selection": "every_input_event_index",
        "split_user_counts": dict(FULL_USER_COUNTS),
        "user_overlap_across_splits": 0,
        "duplicate_event_ids": 0,
    }


@dataclass(frozen=True)
class _FullShardResult:
    """One full-mode input shard rebuilt: its output row and its provenance."""

    split: str
    user_id: str
    shard_row: dict[str, Any]
    provenance_rows: tuple[dict[str, Any], ...]
    fiveshot_planned: dict[str, int]


def _rebuild_full_shard(
    *,
    context: ReplayContext,
    split: str,
    source: InputShard,
    shards_dir: Path,
) -> _FullShardResult:
    """Rebuild every event of one full-mode input shard and write its output."""

    shard = _load_shard(source, signals=True)
    # Material is assigned once per shard rather than per event so the reuse is
    # exactly even; every touch branch then looks its own event up in the plan.
    fiveshot_planned = context.plan_fiveshot_assignment(shard)
    selected: list[int] = []
    signals: dict[int, RebuiltEventSignal] = {}
    provenance_rows: list[dict[str, Any]] = []
    for index in range(shard.event_count):
        try:
            signal, provenance = context.rebuild(shard=shard, index=index)
        except (
            ActionReplayError,
            GenuineTouchRecoveryError,
            KeystrokeReplayError,
            ReplayDatasetBuildError,
        ) as exc:
            raise ReplayDatasetBuildError(
                "full_100k rebuild failed after passing preflight at "
                f"{split}/{source.user_id}/{index}: {exc}"
            ) from exc
        selected.append(index)
        signals[index] = signal
        provenance_rows.append(provenance)
    if selected != list(range(shard.event_count)):
        raise AssertionError("full selection did not keep every index")
    shard_row = _write_output_shard(
        output_path=shards_dir / source.path.name,
        shard=shard,
        selected=selected,
        signals=signals,
        scope="full",
    )
    # Keep peak memory bounded to one user's raw and fake source archives.  All
    # cryptographic bindings are already in the event provenance.
    context.clear_transient_caches()
    return _FullShardResult(
        split=split,
        user_id=str(source.user_id),
        shard_row=shard_row,
        provenance_rows=tuple(provenance_rows),
        fiveshot_planned=dict(fiveshot_planned),
    )


# Populated in the parent immediately before the pool forks.  Workers inherit
# the fully constructed context through the fork, so no part of the build
# configuration is pickled or rebuilt per process.
_FORKED_BUILD_STATE: dict[str, Any] = {}


# Held for the lifetime of each worker process.
_WORKER_THREAD_LIMITS: list[Any] = []


def _forked_worker_initializer() -> None:
    """Pin each worker to a single BLAS thread."""

    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return
    _WORKER_THREAD_LIMITS.append(threadpool_limits(limits=1))


def _forked_shard_task(task: tuple[str, int]) -> _FullShardResult:
    split, order = task
    return _rebuild_full_shard(
        context=_FORKED_BUILD_STATE["context"],
        split=split,
        source=_FORKED_BUILD_STATE["sources"][split][order],
        shards_dir=_FORKED_BUILD_STATE["shards_dir"],
    )


def resolve_build_workers(requested: int | None, *, shard_count: int) -> int:
    """Resolve the requested worker count to the cores this host can spare."""

    if requested is not None and int(requested) > 0:
        return max(1, min(int(requested), shard_count))
    available = os.cpu_count() or 1
    return max(1, min(shard_count, available - 2 if available > 3 else 1))


def _rebuild_full_shards_parallel(
    *,
    context: ReplayContext,
    chosen_sources: Mapping[str, Sequence[InputShard]],
    shards_dir: Path,
    workers: int,
) -> dict[str, list[_FullShardResult]]:
    """Rebuild every full-mode shard across a forked worker pool."""

    _FORKED_BUILD_STATE["context"] = context
    _FORKED_BUILD_STATE["sources"] = {
        split: list(chosen_sources[split]) for split in SPLITS
    }
    _FORKED_BUILD_STATE["shards_dir"] = shards_dir
    tasks = [
        (split, order)
        for split in SPLITS
        for order in range(len(chosen_sources[split]))
    ]
    collected: dict[str, list[_FullShardResult | None]] = {
        split: [None] * len(chosen_sources[split]) for split in SPLITS
    }
    try:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max(1, min(workers, len(tasks))),
            mp_context=multiprocessing.get_context("fork"),
            initializer=_forked_worker_initializer,
        ) as executor:
            futures = {
                executor.submit(_forked_shard_task, task): task for task in tasks
            }
            for future in concurrent.futures.as_completed(futures):
                split, order = futures[future]
                collected[split][order] = future.result()
    finally:
        _FORKED_BUILD_STATE.clear()
    results: dict[str, list[_FullShardResult]] = {}
    for split in SPLITS:
        if any(value is None for value in collected[split]):
            raise ReplayDatasetBuildError(
                f"parallel rebuild did not return every {split} shard"
            )
        results[split] = [
            value for value in collected[split] if value is not None
        ]
    return results


def _build_replay_dataset(
    *,
    input_manifest: str | Path,
    output_dir: str | Path,
    input_release: str | Path | None = None,
    native_genuine_manifest: str | Path | None = None,
    dispatch_quality_ledger: str | Path | None = None,
    raw_trajectory_root: str | Path = DEFAULT_RAW_ROOT,
    split_path: str | Path = DEFAULT_SPLIT_PATH,
    conditional_touch_model: str | Path | None = None,
    conditional_touch_request_model: str | Path | None = None,
    smoke_reference_manifest: str | Path | None = None,
    users_per_split: int = 2,
    events_per_label_action_user: int = 2,
    seed: int = 42,
    full_100k: bool,
    fiveshot_material_dir: str | Path | None = None,
    workers: int | None = None,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    if output.exists():
        raise ReplayDatasetBuildError(f"output already exists: {output}")
    if not full_100k and users_per_split < 2:
        raise ReplayDatasetBuildError("smoke output needs at least two users per split")
    if not full_100k and events_per_label_action_user < 1:
        raise ReplayDatasetBuildError("smoke event quota must be positive")
    if full_100k and smoke_reference_manifest is not None:
        raise ReplayDatasetBuildError(
            "--smoke-reference-manifest is forbidden in full_100k mode"
        )
    if full_100k and conditional_touch_request_model is not None:
        raise ReplayDatasetBuildError(
            "--conditional-touch-request-model is forbidden in full_100k mode"
        )
    if fiveshot_material_dir is None:
        raise ReplayDatasetBuildError(
            "the canonical builder requires --fiveshot-material-dir"
        )
    conditional_touch_model_path: Path | None = None
    conditional_touch_generator: ConditionalTouchGenerator | None = None
    conditional_touch_request_model_path: Path | None = None
    conditional_touch_request_generator: (
        ConditionalTouchRequestGenerator | None
    ) = None
    # The conditional generator is only needed while some action is still
    # dispatched to it.  All three single-pointer actions now transform a
    # five-shot donor instead, so the frozen model becomes optional rather than
    # a hard requirement that would block every build.
    if not full_100k and CONDITIONAL_TOUCH_ACTIONS:
        if conditional_touch_model is None:
            raise ReplayDatasetBuildError(
                "smoke mode requires a frozen --conditional-touch-model artifact"
            )
        conditional_touch_model_path = Path(
            conditional_touch_model
        ).resolve()
        if not conditional_touch_model_path.is_file():
            raise ReplayDatasetBuildError(
                "conditional touch model artifact is missing: "
                f"{conditional_touch_model_path}"
            )
        try:
            conditional_touch_generator = ConditionalTouchGenerator.load(
                conditional_touch_model_path
            )
        except (ConditionalTouchGeneratorError, OSError, ValueError) as exc:
            raise ReplayDatasetBuildError(
                f"conditional touch model artifact is invalid: {exc}"
            ) from exc
        if conditional_touch_request_model is not None:
            conditional_touch_request_model_path = Path(
                conditional_touch_request_model
            ).resolve()
            if not conditional_touch_request_model_path.is_file():
                raise ReplayDatasetBuildError(
                    "conditional touch request model artifact is missing: "
                    f"{conditional_touch_request_model_path}"
                )
            try:
                conditional_touch_request_generator = (
                    ConditionalTouchRequestGenerator.load(
                        conditional_touch_request_model_path
                    )
                )
            except (
                ConditionalTouchRequestGeneratorError,
                OSError,
                ValueError,
            ) as exc:
                raise ReplayDatasetBuildError(
                    f"conditional touch request model artifact is invalid: {exc}"
                ) from exc
    manifest = Path(input_manifest).resolve()
    shards_by_split, input_release_row = load_input_dataset(
        manifest, input_release
    )
    full_input_audit = (
        _validate_full_input_selection(shards_by_split)
        if full_100k
        else None
    )
    native_manifest = Path(
        str(
            native_genuine_manifest
            if native_genuine_manifest is not None
            else input_release_row.get("native_genuine_manifest", "")
        )
    ).resolve()
    quality_ledger = Path(
        str(
            dispatch_quality_ledger
            if dispatch_quality_ledger is not None
            else input_release_row.get("native_data_quality_binding", {}).get(
                "quality_ledger", ""
            )
        )
    ).resolve()
    split_source = Path(split_path).resolve()
    raw_root = Path(raw_trajectory_root).resolve()
    if not native_manifest.is_file() or not quality_ledger.is_file():
        raise ReplayDatasetBuildError("native genuine binding inputs are missing")
    genuine_bindings = load_genuine_touch_bindings(
        native_genuine_manifest=native_manifest,
        dispatch_quality_ledger=quality_ledger,
    )
    (
        selected_genuine_windows,
        selected_genuine_sample_counts,
        genuine_carrier_time_repairs,
    ) = (
        _selected_genuine_windows(shards_by_split, genuine_bindings)
    )
    selected_clusters = set(selected_genuine_windows)
    missing_clusters = selected_clusters - set(genuine_bindings)
    if missing_clusters:
        raise ReplayDatasetBuildError(
            f"selected genuine coverage is incomplete: {len(missing_clusters)}"
        )
    selected_genuine_source_event_ids = {
        genuine_bindings[cluster].source_event_id for cluster in selected_clusters
    }
    duration_sampler = RawWindowRatioSampler.from_selected_train_genuine(
        selected_clusters=selected_clusters,
        window_duration_by_cluster=selected_genuine_windows,
        window_sample_count_by_cluster=selected_genuine_sample_counts,
        genuine_bindings=genuine_bindings,
    )
    template_archive_cache: dict[Path, dict[str, np.ndarray]] = {}
    keystroke_reference_templates = {
        cluster_id: _keystroke_reference_template(
            source_cluster_id=cluster_id,
            genuine_bindings=genuine_bindings,
            raw_archive_cache=template_archive_cache,
        )
        for cluster_id in duration_sampler.keystroke_reference_cluster_ids
    }
    template_archive_cache.clear()
    if len(keystroke_reference_templates) != len(
        duration_sampler.keystroke_reference_cluster_ids
    ):
        raise ReplayDatasetBuildError(
            "compact train-genuine keystroke template coverage is incomplete"
        )
    train_users = _split_train_users(split_source)
    quality_accepted_train_ids = {
        action: {
            binding.source_event_id
            for binding in genuine_bindings.values()
            if binding.split == "train" and binding.action == action
        }
        for action in ACTIONS
    }
    if any(not values for values in quality_accepted_train_ids.values()):
        raise ReplayDatasetBuildError(
            "quality ledger has an empty accepted train action donor pool"
        )
    action_banks: dict[str, ActionReplayBank] = {}
    for action in ("scroll", "swipe", "pinch"):
        bank = ActionReplayBank.from_hmog_npz(
            raw_root / f"hmog_trajectory_{action}.npz",
            train_user_ids=train_users,
            allowed_source_event_ids=quality_accepted_train_ids[action],
            excluded_source_event_ids=selected_genuine_source_event_ids,
            expected_action=action,
        )
        action_banks[action] = bank
    # Fake smoke taps are generated from the frozen conditional model.  No
    # per-event tap donor pool is constructed or consumed at runtime.
    tap_allocators: dict[str, TapReplayAllocator] = {}
    joint_events_root = _joint_events_root(input_release_row)
    chosen_sources = (
        {split: list(shards_by_split[split]) for split in SPLITS}
        if full_100k
        else {
            split: _candidate_users(
                shards_by_split[split], users_per_split=users_per_split
            )
            for split in SPLITS
        }
    )
    smoke_reference_selection: dict[
        SmokeReferenceKey, tuple[str, ...]
    ] | None = None
    smoke_reference_audit: dict[str, Any] | None = None
    smoke_touch_requests: dict[str, SmokeTouchRequest] = {}
    smoke_touch_templates: dict[str, np.ndarray] = {}
    if not full_100k and smoke_reference_manifest is not None:
        (
            smoke_reference_selection,
            smoke_reference_audit,
            smoke_touch_requests,
            smoke_touch_templates,
        ) = _load_smoke_reference_selection(
            reference_manifest=smoke_reference_manifest,
            chosen_sources=chosen_sources,
            events_per_label_action_user=events_per_label_action_user,
        )
    exact_keystroke_event_ids = (
        None
        if smoke_reference_selection is None
        else {
            event_id
            for (split, user_id, action, label), event_ids in (
                smoke_reference_selection.items()
            )
            if action == "keystroke" and label == 1
            for event_id in event_ids
        }
    )
    keystroke_target_plans = _collect_keystroke_target_plans(
        request_shards_by_split=chosen_sources,
        duration_sampler=duration_sampler,
        reference_templates=keystroke_reference_templates,
        joint_events_root=joint_events_root,
        seed=seed,
        request_keystrokes_per_shard=(
            None
            if full_100k or smoke_reference_selection is not None
            else events_per_label_action_user
        ),
        requested_keystroke_event_ids=exact_keystroke_event_ids,
    )
    # The canonical five-shot route transforms the victim's frozen material;
    # it neither reserves raw donor primitives nor allocates generated chunks.
    capacity_audit = {
        "status": "fiveshot_no_donor_allocation",
    }
    action_allocators: dict[tuple[str, str], ReplayAllocator] = {}
    fiveshot_assignment_audit: dict[str, dict[str, int]] = {}
    # Assignment is planned over every fake event the shard holds, not over
    # the subset a small build rebuilds, so the cap is the release's own
    # per-group size in both modes.
    requested_per_group = FIVESHOT_FAKE_EVENTS_PER_GROUP
    try:
        fiveshot_material = load_fiveshot_material(
            fiveshot_material_dir,
            maximum_uses_per_shot=max(
                1, int(math.ceil(requested_per_group / FIVESHOT_SHOTS_PER_GROUP))
            ),
            genuine_touch_bindings=genuine_bindings,
        )
    except FiveShotMaterialError as exc:
        raise ReplayDatasetBuildError(
            f"five-shot material could not be loaded: {exc}"
        ) from exc

    context = ReplayContext(
        action_banks=action_banks,
        action_allocators=action_allocators,
        tap_allocators=tap_allocators,
        keystroke_target_plans=keystroke_target_plans,
        genuine_bindings=genuine_bindings,
        duration_sampler=duration_sampler,
        joint_events_root=joint_events_root,
        seed=seed,
        # A five-shot build takes every fake tap from the victim's own frozen
        # events, so the conditional generator is not part of that route no
        # matter which size the build is.
        tap_strategy="bound_fake_tap_android_zoh",
        conditional_touch_generator=conditional_touch_generator,
        conditional_touch_model_path=conditional_touch_model_path,
        conditional_touch_request_generator=(
            conditional_touch_request_generator
        ),
        conditional_touch_request_model_path=(
            conditional_touch_request_model_path
        ),
        smoke_touch_requests=smoke_touch_requests,
        smoke_touch_templates=smoke_touch_templates,
        fiveshot_material=fiveshot_material,
    )
    output.mkdir(parents=True)
    shards_dir = output / "shards"
    shards_dir.mkdir()
    provenance_rows: list[dict[str, Any]] = []
    provenance_path = output / "provenance.jsonl"
    provenance_handle = (
        provenance_path.open("x", encoding="utf-8") if full_100k else None
    )
    action_label_counts: dict[str, int] = {}
    output_genuine_carrier_time_repairs = 0

    def record_provenance(row: dict[str, Any]) -> None:
        nonlocal output_genuine_carrier_time_repairs
        label_name = "genuine" if int(row["label"]) == 0 else "fake"
        key = f"{row['split']}/{row['action']}/{label_name}"
        action_label_counts[key] = action_label_counts.get(key, 0) + 1
        if (
            int(row["label"]) == 0
            and row.get("donor", {}).get(
                "carrier_time_repaired_from_genuine_binding"
            )
            is True
        ):
            output_genuine_carrier_time_repairs += 1
        if provenance_handle is None:
            provenance_rows.append(row)
        else:
            provenance_handle.write(json.dumps(row, sort_keys=True) + "\n")

    manifest_rows: list[dict[str, Any]] = []
    rejection_counts: dict[str, int] = {}
    observed_smoke_reference_selection: dict[
        SmokeReferenceKey, tuple[str, ...]
    ] = {}
    # Every full-mode shard is independent, so they are rebuilt across a forked
    # worker pool and then absorbed in the original source order.  The results
    # are ordered, not raced, so provenance and manifest rows land in the exact
    # sequence a single-process build produced.
    build_workers = (
        resolve_build_workers(
            workers,
            shard_count=sum(len(chosen_sources[split]) for split in SPLITS),
        )
        if full_100k
        else 1
    )
    parallel_results: dict[str, list[_FullShardResult]] | None = None
    if full_100k and build_workers > 1:
        parallel_results = _rebuild_full_shards_parallel(
            context=context,
            chosen_sources=chosen_sources,
            shards_dir=shards_dir,
            workers=build_workers,
        )
    for split in SPLITS:
        output_shards: list[dict[str, Any]] = []
        if full_100k:
            full_results = (
                parallel_results[split]
                if parallel_results is not None
                else [
                    _rebuild_full_shard(
                        context=context,
                        split=split,
                        source=source,
                        shards_dir=shards_dir,
                    )
                    for source in chosen_sources[split]
                ]
            )
            for result in full_results:
                if result.fiveshot_planned:
                    fiveshot_assignment_audit[f"{split}/{result.user_id}"] = {
                        action: int(count)
                        for action, count in sorted(
                            result.fiveshot_planned.items()
                        )
                    }
                for row in result.provenance_rows:
                    record_provenance(row)
                output_shards.append(result.shard_row)
        for source in () if full_100k else chosen_sources[split]:
            shard = _load_shard(source, signals=True)
            labels = np.asarray(shard.arrays["label"], dtype=np.int64)
            actions = np.asarray(shard.arrays["action"]).astype(str)
            # Material is assigned once per shard rather than per event so the
            # reuse is exactly even; every touch branch then looks its own event
            # up in the resulting plan.
            fiveshot_planned = context.plan_fiveshot_assignment(shard)
            if fiveshot_planned:
                fiveshot_assignment_audit[f"{split}/{source.user_id}"] = {
                    action: int(count)
                    for action, count in sorted(fiveshot_planned.items())
                }
            selected: list[int] = []
            signals: dict[int, RebuiltEventSignal] = {}
            if smoke_reference_selection is not None:
                (
                    selected,
                    signals,
                    observed_for_shard,
                ) = _rebuild_reference_locked_smoke_shard(
                    context=context,
                    shard=shard,
                    selection=smoke_reference_selection,
                    record_provenance=record_provenance,
                )
                overlap = set(observed_smoke_reference_selection).intersection(
                    observed_for_shard
                )
                if overlap:
                    raise ReplayDatasetBuildError(
                        "reference-locked smoke groups were rebuilt twice"
                    )
                observed_smoke_reference_selection.update(observed_for_shard)
            else:
                for action in ACTIONS:
                    for label in (0, 1):
                        candidates = np.flatnonzero(
                            (actions == action) & (labels == label)
                        )
                        accepted = 0
                        last_error: Exception | None = None
                        for index in candidates:
                            try:
                                signal, provenance = context.rebuild(
                                    shard=shard, index=int(index)
                                )
                            except (
                                ActionReplayError,
                                GenuineTouchRecoveryError,
                                KeystrokeReplayError,
                                ReplayDatasetBuildError,
                            ) as exc:
                                last_error = exc
                                key = (
                                    f"{split}/{action}/{label}/"
                                    f"{type(exc).__name__}"
                                )
                                rejection_counts[key] = (
                                    rejection_counts.get(key, 0) + 1
                                )
                                continue
                            selected.append(int(index))
                            signals[int(index)] = signal
                            record_provenance(provenance)
                            accepted += 1
                            if accepted == events_per_label_action_user:
                                break
                        if accepted != events_per_label_action_user:
                            raise ReplayDatasetBuildError(
                                f"{source.user_id}/{action}/{label}: only rebuilt "
                                f"{accepted}/{events_per_label_action_user}; "
                                f"last error: {last_error}"
                            )
            output_shards.append(
                _write_output_shard(
                    output_path=shards_dir / source.path.name,
                    shard=shard,
                    selected=selected,
                    signals=signals,
                    scope=SMOKE_MANIFEST_SCOPE,
                )
            )
            context.clear_transient_caches()
        events = sum(int(row["events"]) for row in output_shards)
        fake = sum(int(row["fake"]) for row in output_shards)
        genuine = sum(int(row["genuine"]) for row in output_shards)
        manifest_rows.append(
            {
                "schema_version": INPUT_MANIFEST_SCHEMA,
                "scope": "full" if full_100k else SMOKE_MANIFEST_SCOPE,
                "formal_result": False,
                "split": split,
                "events": events,
                "fake_events": fake,
                "genuine_events": genuine,
                "user_ids": [row["user_id"] for row in output_shards],
                "shards": output_shards,
            }
        )
    if smoke_reference_selection is not None:
        if smoke_reference_audit is None:
            raise ReplayDatasetBuildError(
                "smoke reference audit state is missing"
            )
        smoke_reference_audit = _finalize_smoke_reference_audit(
            expected=smoke_reference_selection,
            observed=observed_smoke_reference_selection,
            audit=smoke_reference_audit,
        )
    if full_100k:
        observed_fake = sum(int(row["fake_events"]) for row in manifest_rows)
        observed_genuine = sum(
            int(row["genuine_events"]) for row in manifest_rows
        )
        if (
            full_input_audit is None
            or observed_fake != 100_000
            or observed_fake != int(full_input_audit["fake_events"])
            or observed_genuine != int(full_input_audit["genuine_events"])
        ):
            raise ReplayDatasetBuildError(
                "full output did not preserve the complete input selection"
            )
    if provenance_handle is not None:
        provenance_handle.close()
    else:
        with provenance_path.open("x", encoding="utf-8") as handle:
            for row in sorted(
                provenance_rows,
                key=lambda value: (
                    SPLITS.index(str(value["split"])),
                    str(value["user_id"]),
                    str(value["event_id"]),
                ),
            ):
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    event_manifest = output / "event_manifest.jsonl"
    with event_manifest.open("x", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    audit_counts = _validate_donor_provenance(
        (
            _read_jsonl_objects(provenance_path)
            if full_100k
            else provenance_rows
        ),
        selected_genuine_source_event_ids=selected_genuine_source_event_ids,
    )
    expected_conditional_touch_events = (
        0
        if full_100k or fiveshot_material is not None
        else sum(
            int(action_label_counts.get(f"{split}/{action}/fake", 0))
            for split in SPLITS
            for action in CONDITIONAL_TOUCH_ACTIONS
        )
    )
    # A five-shot build transforms the victim's own frozen tap/scroll/swipe in
    # both sizes, so every fake event of those three actions carries the exact
    # template method.  Only the legacy full route, which replayed raw donors
    # instead, generated none.
    expected_exact_touch_events = (
        0
        if full_100k and fiveshot_material is None
        else sum(
            int(action_label_counts.get(f"{split}/{action}/fake", 0))
            for split in SPLITS
            for action in EXACT_TOUCH_TEMPLATE_ACTIONS
        )
    )
    if audit_counts["exact_touch_template_generated_events"] != (
        expected_exact_touch_events
    ):
        raise ReplayDatasetBuildError(
            "exact touch provenance coverage differs from the fake selection"
        )
    if audit_counts["conditional_touch_generated_events"] != (
        expected_conditional_touch_events
    ):
        raise ReplayDatasetBuildError(
            "conditional touch provenance coverage differs from smoke selection"
        )
    # Exercise the exact detector loader before signing the release.
    loaded_rows = _load_manifest(event_manifest)
    for split in SPLITS:
        load_event_partition(loaded_rows[split])
    fake_events = sum(int(row["fake_events"]) for row in manifest_rows)
    genuine_events = sum(int(row["genuine_events"]) for row in manifest_rows)
    if (
        full_100k
        and full_input_audit is not None
        and dict(sorted(action_label_counts.items()))
        != full_input_audit["action_label_counts"]
    ):
        raise ReplayDatasetBuildError(
            "full output action/label counts differ from the complete input"
        )
    mode = "full_100k" if full_100k else "smoke"
    scope = "full" if full_100k else SMOKE_MANIFEST_SCOPE
    # Five-shot taps are transforms of the victim's own frozen tap through the
    # exact template route in both sizes, which is what the provenance records.
    tap_method = (
        "bound_fake_tap_android_zoh"
        if full_100k and fiveshot_material is None
        else EXACT_TOUCH_TEMPLATE_REBUILD_METHOD
    )
    release = {
        "schema_version": DATASET_RELEASE_SCHEMA,
        "status": "ready",
        "mode": mode,
        "manifest_scope": scope,
        "formal_result": False,
        "events": fake_events + genuine_events,
        "fake_events": fake_events,
        "genuine_events": genuine_events,
        "fake_selection": (
            "all_100000"
            if full_100k
            else (
                "smoke_exact_template_reference_locked"
                if smoke_reference_audit is not None
                else "smoke_conditional_touch_generation"
            )
        ),
        "smoke_reference_selection": (
            {
                "enabled": False,
                "status": "not_requested",
            }
            if smoke_reference_audit is None
            else {
                "enabled": True,
                **smoke_reference_audit,
            }
        ),
        "full_100k_supported": bool(full_100k),
        "users": 100 if full_100k else users_per_split * len(SPLITS),
        "actions": list(ACTIONS),
        "modalities": list(DETECTOR_MODALITIES),
        "split_policy": "fixed_user_disjoint_70_10_20",
        "storage": (
            "one compressed ragged full shard per each of 100 users"
            if full_100k
            else "one compressed ragged smoke shard per selected user"
        ),
        "event_manifest": str(event_manifest),
        "event_manifest_sha256": sha256_file(event_manifest),
        "provenance": str(provenance_path),
        "provenance_sha256": sha256_file(provenance_path),
        "action_label_counts": dict(sorted(action_label_counts.items())),
        "acceptance_audit_used": False,
        "clean_candidate_filter_used": False,
        "pass_binding_used": False,
        "six_stage_supervisor_used": False,
        "trajectory_rebuild": {
            "schema_version": REBUILDER_SCHEMA,
            "input_manifest": str(manifest),
            "input_manifest_sha256": sha256_file(manifest),
            "input_release_sha256": sha256_file(
                manifest.parent / "release.json"
                if input_release is None
                else Path(input_release).resolve()
            ),
            "native_genuine_manifest": str(native_manifest),
            "native_genuine_manifest_sha256": sha256_file(native_manifest),
            "dispatch_quality_ledger": str(quality_ledger),
            "dispatch_quality_ledger_sha256": sha256_file(quality_ledger),
            "split_path": str(split_source),
            "split_path_sha256": sha256_file(split_source),
            "raw_trajectory_root": str(raw_root),
            "common_observer": "common_android_zoh_v1",
            "genuine_method": "raw_genuine_recovery",
            # The attacker's five recordings are drawn from the same archive and
            # by the same observer as the genuine class, so neither side of the
            # comparison carries a rendering the other does not.
            "fiveshot_material_touch_rendering": (
                "raw_genuine_recovery"
                if fiveshot_material is not None
                else None
            ),
            "fiveshot_donor_pairing": (
                None
                if fiveshot_material is None
                else "tap_scroll_swipe=carrier_sample_count;pinch=area_span"
            ),
            # The three switches that decide how an action is generated are read
            # from the environment at import, so a release that did not record
            "generation_policy_switches": {
                "HMOG_SYNTHESISED_CLOCK_ACTIONS": sorted(
                    SYNTHESISED_CLOCK_ACTIONS
                ),
                "HMOG_FIVESHOT_TIMING_ACTIONS": sorted(FIVESHOT_TIMING_ACTIONS),
                "HMOG_FIVESHOT_TIMING_SPREAD": FIVESHOT_TIMING_SPREAD,
            },
            "tap_method": tap_method,
            "tap_method_note": (
                "Five-shot mode transforms one of the victim's own five frozen "
                "real taps onto the fake event's bound Android target and "
                "applies the common ZOH observer.  The target's DOWN row is the "
                "exact request anchor; its UP row is not, because every "
                "generated tap target reports the same coordinate twice and "
                "requesting that would delete the donor's own lift-off drift.  "
                "The request carries the donor chord instead, bounded by the "
                f"carrier touch slop of {FIVESHOT_TAP_DRIFT_LIMIT_PX:g} px."
                if fiveshot_material is not None
                else "Full mode retains each fake tap's own generated Android "
                "target binding and applies the common ZOH observer; it is not "
                "human tap replay."
                if full_100k
                else (
                    "Smoke mode calls the independent exact-touch interface "
                    "with a frozen donor template and authoritative start, end, "
                    "direction, and duration; identity requests are byte-stable."
                )
            ),
            # Under five-shot every one of these three actions is a transform
            # of the victim's own frozen event, so neither the legacy raw-donor
            # wording nor the smoke conditional wording describes it.
            "scroll_swipe_pinch_method": (
                f"scroll_swipe={EXACT_TOUCH_TEMPLATE_REBUILD_METHOD};"
                f"pinch={PINCH_AREA_SIMILARITY_REBUILD_METHOD}"
                if fiveshot_material is not None
                else (
                    "train_raw_action_endpoint_controlled_replay"
                    if full_100k
                    else (
                        "conditional_scroll_plus_exact_template_swipe_plus_"
                        "train_raw_pinch_endpoint_replay"
                    )
                )
            ),
            "scroll_swipe_pinch_spatial_policy": (
                "scroll_swipe_exact_bound_target_endpoints;"
                "pinch_similarity_at_widest_spread_no_scale_gate"
                if fiveshot_material is not None
                else (
                    (
                        "scroll_swipe_train_real_D4_reference;"
                        if full_100k
                        else (
                            "scroll_train_fitted_conditional_exact_endpoints;"
                            "swipe_independent_exact_template_chord_frame;"
                        )
                    )
                    + "pinch_exact_live_pointer_endpoints_with_0.80_1.25_bounded_residual"
                )
            ),
            "scroll_swipe_pinch_geometry_preserved": (
                "scroll_swipe_exact_start_end_direction_duration;"
                "pinch_exact_widest_two_pointer_endpoints"
                if fiveshot_material is not None
                else (
                    (
                        "scroll_swipe_exact_train_real_start_endpoint_direction;"
                        if full_100k
                        else "scroll_swipe_exact_start_end_direction_duration;"
                    )
                    + "pinch_exact_four_live_pointer_endpoints_and_in_out"
                )
            ),
            "scroll_swipe_method": (
                EXACT_TOUCH_TEMPLATE_REBUILD_METHOD
                if fiveshot_material is not None
                else (
                    "train_raw_action_isometric_replay"
                    if full_100k
                    else (
                        "scroll=" + CONDITIONAL_TOUCH_REBUILD_METHOD
                        + ";swipe=" + EXACT_TOUCH_TEMPLATE_REBUILD_METHOD
                    )
                )
            ),
            # A five-shot build transforms the victim's own frozen tap, scroll
            # and swipe through the exact template route in both sizes, so the
            # release states that it ran rather than repeating the legacy full
            # route's claim that it never does.
            "exact_touch_template_generator_used": bool(
                not full_100k or fiveshot_material is not None
            ),
            "exact_touch_template_actions": (
                []
                if full_100k and fiveshot_material is None
                else sorted(EXACT_TOUCH_TEMPLATE_ACTIONS)
            ),
            "exact_touch_template_generator_schema_version": (
                None
                if full_100k and fiveshot_material is None
                else EXACT_TOUCH_TEMPLATE_SCHEMA_VERSION
            ),
            "conditional_touch_generator_used": bool(not full_100k),
            "conditional_touch_actions": (
                [] if full_100k else sorted(CONDITIONAL_TOUCH_ACTIONS)
            ),
            "conditional_touch_model": (
                None
                if conditional_touch_model_path is None
                else str(conditional_touch_model_path)
            ),
            "conditional_touch_model_file_sha256": (
                context.conditional_touch_model_file_sha256
            ),
            "conditional_touch_model_canonical_artifact_sha256": (
                context.conditional_touch_model_artifact_sha256
            ),
            "conditional_touch_model_schema_version": (
                None
                if conditional_touch_generator is None
                else conditional_touch_generator.schema_version
            ),
            "conditional_touch_training_summary": (
                context.conditional_touch_training_summary
            ),
            "conditional_touch_training_summary_sha256": (
                context.conditional_touch_training_summary_sha256
            ),
            "conditional_touch_request_model": (
                None
                if conditional_touch_request_model_path is None
                else str(conditional_touch_request_model_path)
            ),
            "conditional_touch_request_model_file_sha256": (
                context.conditional_touch_request_model_file_sha256
            ),
            "conditional_touch_request_model_canonical_artifact_sha256": (
                context.conditional_touch_request_model_artifact_sha256
            ),
            "conditional_touch_request_model_schema_version": (
                None
                if conditional_touch_request_generator is None
                else conditional_touch_request_generator.schema_version
            ),
            "conditional_touch_request_training_summary": (
                context.conditional_touch_request_training_summary
            ),
            "conditional_touch_conditioning": (
                None
                if full_100k
                else (
                    "train_fitted_request_geometry+bound_raw_timestamps+"
                    "event_seed"
                    if conditional_touch_request_generator is not None
                    else "action+orientation+exact_DOWN_start+exact_UP_end+"
                    "direction+bound_raw_timestamps+event_seed"
                )
            ),
            "conditional_touch_request_identity_role": (
                None
                if conditional_touch_request_generator is None
                else "carrier_only"
            ),
            "conditional_touch_request_plan_replay_semantics": (
                None
                if conditional_touch_request_generator is None
                else "not_original_plan_replay"
            ),
            "conditional_touch_runtime_donor_lookup_used": False,
            "conditional_touch_output_imu_policy": (
                None if full_100k else "retain_input_fake_event_imu_unchanged"
            ),
            "conditional_touch_coordinate_clipping_used": False,
            "conditional_touch_endpoint_audit_tolerance_px": (
                None if full_100k else 5.0e-4
            ),
            # Five-shot pinch is one similarity transform of the victim's own
            # frozen event anchored at its widest-spread moment; it has no
            # donor bank and therefore no scale gate at all, so repeating the
            # legacy method name and its [0.80, 1.25] bounds would misdescribe
            # the release.
            "pinch_method": (
                PINCH_AREA_SIMILARITY_REBUILD_METHOD
                if fiveshot_material is not None
                else "train_raw_action_pinch_endpoint_replay"
            ),
            "pinch_coordinate_clipping_used": False,
            "pinch_center_and_span_scale_bounds": (
                None if fiveshot_material is not None else [0.80, 1.25]
            ),
            # Five-shot typing composes the victim's own hold/flight rhythm onto
            # the carrier's key anchors and drives the IMU from that same key
            # schedule with a per-victim impulse adapter.
            "keystroke_method": (
                FIVESHOT_KEYSTROKE_TOUCH_METHOD
                + "+fiveshot_keystroke_imu_pulse"
            ),
            "keystroke_touch_method": FIVESHOT_KEYSTROKE_TOUCH_METHOD,
            "keystroke_target_replacement": (
                "one_complete_coupled_selected_train_genuine_carrier"
            ),
            "keystroke_generated_imu_method": "fiveshot_keystroke_imu_pulse",
            "linear_touch_interpolation_used": False,
            "coordinate_clipping_used": False,
            "output_split_disjoint_touch_donor_families": True,
            "selected_genuine_touch_source_exclusion": True,
            "touch_replay_quality_accepted_train_donors_only": True,
            "quality_accepted_train_donor_counts": {
                action: len(values)
                for action, values in quality_accepted_train_ids.items()
            },
            "raw_to_window_duration_sampler": duration_sampler.summary(),
            "action_update_rate_matching": (
                "same_deterministically_sampled_train_genuine_timing_reference"
            ),
            "maximum_action_time_warp": 1.0,
            "duration_binding": "retained_trajectory_elapsed_time_endpoint",
            "duration_binding_note": (
                "The retained trajectory elapsed-time endpoint determines the "
                "observable window independently of sample count. Genuine all-zero "
                "carrier time is repaired from the exact quality-bound raw archive. "
                "For fake keystrokes, one complete train-genuine carrier jointly "
                "binds key sequence, raw touch duration, detector duration and output "
                "sample count. The victim's five-shot pulse model then drives IMU "
                "impacts from the same key-down schedule as the touch carrier."
            ),
            "genuine_carrier_time_repairs": (
                output_genuine_carrier_time_repairs
            ),
            "input_genuine_carrier_time_repairs_identified": (
                genuine_carrier_time_repairs
            ),
            "genuine_carrier_time_recovery_source": (
                "quality_bound_raw_archive_touch_duration_ms"
            ),
            "keystroke_timing_support": "robust_train_5_95_percentile",
            "donor_audit": audit_counts,
            "rejection_counts": dict(sorted(rejection_counts.items())),
            "full_input_selection_audit": full_input_audit,
            "full_replay_capacity_audit": capacity_audit,
            "full_100k_supported": bool(full_100k),
            "full_100k_blocker": None if full_100k else (
                "not evaluated in smoke mode; use explicit --full-100k for the "
                "exact preflight and complete selection"
            ),
        },
    }
    _write_json(output / "release.json", release)
    return release


def build_smoke_replay_dataset(
    *,
    input_manifest: str | Path,
    output_dir: str | Path,
    input_release: str | Path | None = None,
    native_genuine_manifest: str | Path | None = None,
    dispatch_quality_ledger: str | Path | None = None,
    raw_trajectory_root: str | Path = DEFAULT_RAW_ROOT,
    split_path: str | Path = DEFAULT_SPLIT_PATH,
    conditional_touch_model: str | Path | None = None,
    conditional_touch_request_model: str | Path | None = None,
    smoke_reference_manifest: str | Path | None = None,
    users_per_split: int = 2,
    events_per_label_action_user: int = 2,
    seed: int = 42,
    fiveshot_material_dir: str | Path | None = None,
) -> dict[str, Any]:
    return _build_replay_dataset(
        input_manifest=input_manifest,
        output_dir=output_dir,
        input_release=input_release,
        native_genuine_manifest=native_genuine_manifest,
        dispatch_quality_ledger=dispatch_quality_ledger,
        raw_trajectory_root=raw_trajectory_root,
        split_path=split_path,
        conditional_touch_model=conditional_touch_model,
        conditional_touch_request_model=conditional_touch_request_model,
        smoke_reference_manifest=smoke_reference_manifest,
        users_per_split=users_per_split,
        events_per_label_action_user=events_per_label_action_user,
        seed=seed,
        full_100k=False,
        fiveshot_material_dir=fiveshot_material_dir,
    )


def build_full_100k_replay_dataset(
    *,
    input_manifest: str | Path,
    output_dir: str | Path,
    input_release: str | Path | None = None,
    native_genuine_manifest: str | Path | None = None,
    dispatch_quality_ledger: str | Path | None = None,
    raw_trajectory_root: str | Path = DEFAULT_RAW_ROOT,
    split_path: str | Path = DEFAULT_SPLIT_PATH,
    smoke_reference_manifest: str | Path | None = None,
    seed: int = 42,
    fiveshot_material_dir: str | Path | None = None,
    workers: int | None = None,
) -> dict[str, Any]:
    """Build all input events after exact full donor/target preflight."""

    return _build_replay_dataset(
        input_manifest=input_manifest,
        output_dir=output_dir,
        input_release=input_release,
        native_genuine_manifest=native_genuine_manifest,
        dispatch_quality_ledger=dispatch_quality_ledger,
        raw_trajectory_root=raw_trajectory_root,
        split_path=split_path,
        smoke_reference_manifest=smoke_reference_manifest,
        seed=seed,
        full_100k=True,
        fiveshot_material_dir=fiveshot_material_dir,
        workers=workers,
    )


__all__ = [
    "AndroidTarget",
    "DETECTOR_MODALITIES",
    "DurationRatioSample",
    "RawWindowRatioSampler",
    "ROBUST_KEYSTROKE_BOUNDS",
    "ReplayDatasetBuildError",
    "SMOKE_MANIFEST_SCOPE",
    "TapReplayAllocator",
    "TapDonor",
    "build_full_100k_replay_dataset",
    "build_smoke_replay_dataset",
    "load_android_target",
    "observe_bound_android_target",
    "load_input_dataset",
]
