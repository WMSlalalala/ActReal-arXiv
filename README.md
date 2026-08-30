# ActReal: arXiv Artifact

This is the public artifact release accompanying the ActReal arXiv preprint.

It contains the implementation, frozen checkpoints, one final study APK, and
de-identified processed phone records from 20 participants.

The paper-to-artifact index is
[`USENIX_SECURITY_2027_ARTIFACT_MAP.csv`](USENIX_SECURITY_2027_ARTIFACT_MAP.csv).

## Contents

| Path | Contents |
| --- | --- |
| `methods/` | Touch and IMU generation, Android delivery and hooks, study-app source and APK, six detector implementations, and evaluation code. |
| `checkpoints/detectors/` | 90 frozen detector models and their 90 development-selected thresholds: five actions by three modalities by six detectors. |
| `checkpoints/generator/` | 24 frozen gesture-generator models and 24 matching effective configurations, covering the released five-shot, zero-shot, and component-ablation protocols. |
| `data/on_device/` | 400 processed NPZ archives from 20 participants on a Pixel 10 and Galaxy S21, plus collection and file inventories. |
| `data/splits/users_seed42.json` | Frozen HMOG user identifiers only; no HMOG signals. |
| `data/event_level/ACTION_BUNDLE_MAP.json` | Provenance labels for licensed event-level inputs. |
| `media/demos/` | Two short, silent demonstrations of the agent-execution and human-interaction workflows. |

## Demonstration videos

The following clips demonstrate the two complete real-device workflows. Audio
and source metadata, including location, capture time, and device information,
have been removed.

### ActReal agent execution

https://github.com/user-attachments/assets/1ec4b2c5-6aae-47a7-9485-c9a6c35b5652

[Play from the repository mirror](media/demos/actreal_agent_execution.mp4)

The agent carries out a multi-step shopping task on the research phone. ActReal
normalizes each semantic action, realizes it as touch and IMU events, and
delivers both streams while the live monitor displays the touch trajectory,
accelerometer, gyroscope, and contact signals. To make the injection process
easier to inspect, the demonstration highlights each injected action and
briefly pauses the workflow after the injection.

### Human interaction

https://github.com/user-attachments/assets/4480b5d6-7f35-4400-88aa-ff409482a245

[Play from the repository mirror](media/demos/human_interaction.mp4)

A participant performs a shopping interaction manually while the collector
records synchronized touch and six-axis IMU events and the live monitor
displays the captured streams.

The demonstrated agent-side flow is:

1. The agent produces a semantic action.
2. ActReal normalizes it into one of the five supported physical actions.
3. ActReal places touch and IMU events on a common `ActionBundle` timeline and
   delivers them through the Android event paths.
4. The live monitor displays the application-visible touch and six-axis IMU
   signals.

## Setup and run

Use Python 3.10 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Generate a Tap IMU sequence for participant `P00` on the Pixel 10 using the
released five-shot checkpoint and the full 240-step sampling configuration:

```bash
python methods/generation/imu/scripts/generate_imu.py tap \
  --x 540 --y 960 --user-id 0 --reference-device pixel10 \
  --duration-ms 100 --sample-steps 240 --seed 42 \
  --out generated/tap_p00_pixel10.npz
```

Public user IDs `0` through `19` map to `P00` through `P19`. Select either
`pixel10` or `s21`; references from different participant-device records are
not mixed. Tap, Swipe, and Pinch use 240 sampling steps, while Scroll uses 320.
The action-specific geometry arguments are listed by:

```bash
python methods/generation/imu/scripts/generate_imu.py --help
```

Install the released study APK on an authorized Android device with:

```bash
adb install -r methods/app/apk/SensorStudyCollector.apk
```

The frozen-detector real-device scorer takes separately authorized Pixel and
Galaxy ActReal session roots:

```bash
python methods/evaluation/on_device/score_matrix_far.py \
  --pixel-root <pixel-session-root> \
  --s21-root <galaxy-session-root> \
  --out <output-directory>
```

Those agent execution sessions are not included in this artifact. The released
processed participant archives can be opened directly with NumPy, for example:

```bash
python -c "import numpy as np; z=np.load('data/on_device/P00/pixel10/test/tap.npz', allow_pickle=False); print(z.files)"
```

## Main entry points

| Purpose | Entry point |
| --- | --- |
| Export processed phone records from authorized raw sessions | `methods/evaluation/on_device/export_processed_phone_data.py` |
| Generate Tap, Scroll, Swipe, or Pinch IMU | `methods/generation/imu/scripts/generate_imu.py` |
| Train an action-specific diffusion model | `methods/generation/imu/diffusion_model/train.py` |
| Load all frozen detector cells | `methods/evaluation/event_level/common/release_cell_map.json` |
| Score authorized real-device sessions with frozen FAR thresholds | `methods/evaluation/on_device/score_matrix_far.py` |
| Build the local Android study application | `methods/app/source/` |
| Install the final study application | `methods/app/apk/SensorStudyCollector.apk` |

## Processed human data

`data/on_device/` contains only de-identified, event-aligned derivatives. Each
participant completed the three local controlled tasks on both phones and a
five-shot calibration on each phone. The release therefore contains:

- 20 participants (`P00` through `P19`);
- 40 participant-device records;
- 120 canonical task runs;
- 400 NPZ files: 20 participants by two phones by two splits by five actions;
- 1,000 calibration events and 3,608 task events.

The frozen task-event totals are 1,573 Tap, 1,141 Scroll, 127 Swipe, 167 Pinch,
and 600 Keystroke events. `data/on_device/inventory.json` stores the collection
summary, and `data/on_device/inventory.csv` stores the sample counts, array
counts, and byte sizes for every NPZ.

The directory layout is:

```text
data/on_device/
  inventory.json
  inventory.csv
  P00/ ... P19/
    pixel10/
      fewshot/{tap,scroll,swipe,pinch,keystroke}.npz
      test/{tap,scroll,swipe,pinch,keystroke}.npz
    s21/
      fewshot/{tap,scroll,swipe,pinch,keystroke}.npz
      test/{tap,scroll,swipe,pinch,keystroke}.npz
```

Every archive uses schema `actreal_on_device_processed_v1` and can be loaded
with `numpy.load(..., allow_pickle=False)`. Common fields include archive
descriptors, event counts, six-axis 100 Hz IMU arrays, validity and action
masks, active lengths, durations, orientation, normalized endpoints, redacted
Keystroke counts, and ragged touch arrays with relative event times. The fixed
IMU window lengths are 35, 179, 167, 116, and 256 frames for Tap, Scroll,
Swipe, Pinch, and Keystroke, respectively.

For Tap, Scroll, Swipe, and Pinch, the touch arrays contain normalized pointer
coordinates and non-identifying MotionEvent fields. For Keystroke, the release
stores only redacted edit timing and before/after/add/remove counts. It stores
no key labels, typed text, screen coordinates, wall-clock timestamps, names,
contact details, audio, video, screenshots, accessibility text, or application
content.

## Checkpoints and external inputs

Detector directories follow:

```text
checkpoints/detectors/<action>__<modality>__<detector>/
```

Actions are Tap, Scroll, Swipe, Pinch, and Keystroke. Modalities are touch
trajectory (`trajectory_xytime`), six-axis IMU (`imu_only`), and joint
touch-IMU (`imu_trajectory_xytime`). Classical models use `model.joblib`, deep
models use `checkpoint.pt`, and every cell includes `thresholds.json`.

Generator directories follow:

```text
checkpoints/generator/five_shot/<action>/
checkpoints/generator/zero_shot/<action>/
checkpoints/generator/component_ablation/<arm>/<action>/
```

Keystroke uses the released non-diffusion adapter and therefore has no
diffusion checkpoint. The `.pt` files are inference artifacts and do not
contain optimizer state.

## License

ActReal-authored code is licensed under the Apache License 2.0. Third-party
components remain subject to the licenses included in their respective
directories.
