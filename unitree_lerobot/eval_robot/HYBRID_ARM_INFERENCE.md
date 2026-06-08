# Hybrid Arm Inference

This mode is intended for the temporary hardware setup where a G1-29 robot is
available but the four training cameras and Dex3 hands are not.

It builds each ACT observation as:

```text
four images from the selected dataset frame
+ real G1 dual-arm state                 [14]
+ Dex3 state from the selected dataset   [14]
= policy observation.state               [28]
```

ACT predicts a full action chunk with shape `[chunk_size, 28]`. The runner only
uses the first 14 dimensions:

```text
predicted action chunk [T, 28]
                  -> arm targets [T, 14]
                  -> per-joint delta limit
                  -> G1 dual-arm controller
```

The dataset cursor advances once for every action consumed. Inference runs in
a single background worker while the main thread continues publishing queued
actions at the requested frequency. When the queue falls below
`--prefetch-threshold` of `--actions-per-inference`, the runner snapshots the
latest real arm state and dataset observation and asks ACT for a new chunk.

Actions are keyed by dataset timestep. When a newer chunk arrives, stale
actions are discarded, overlapping future actions are replaced, and the
remaining horizon is appended. If inference misses the available queue
horizon, the runner holds the latest target while waiting instead of executing
an invalid action.

This is not a real visual closed loop. The policy sees recorded images while
the arm state comes from the real robot. Use it only to validate the model,
joint mapping, DDS connection, and arm trajectory behavior.

## Dry Run

Dry-run is the default. It subscribes to `rt/lowstate`, performs hybrid
inference, and writes logs without creating an arm command publisher.

```bash
python unitree_lerobot/eval_robot/hybrid_arm_infer.py \
  --policy-path=/path/to/checkpoint/pretrained_model \
  --episode=0 \
  --max-policy-steps=30 \
  --actions-per-inference=5 \
  --network-interface=eth0
```

Inspect the generated `hybrid_arm_results/<run>/steps.csv` before enabling
control. It contains the real state, dataset state, raw prediction, limited
target, dataset action, and timing for every policy step.

## Real Arm Control

Real control requires both `--send-actions` and an explicit confirmation
value. Predicted targets are limited relative to the latest measured state by
`--max-action-delta-rad`.

Start with a small step count and a conservative delta:

```bash
python unitree_lerobot/eval_robot/hybrid_arm_infer.py \
  --policy-path=/path/to/checkpoint/pretrained_model \
  --episode=0 \
  --max-policy-steps=10 \
  --actions-per-inference=1 \
  --max-action-delta-rad=0.02 \
  --network-interface=eth0 \
  --send-actions \
  --control-confirmation=SEND_TO_REAL_G1
```

To move slowly toward the arm pose at the selected dataset frame before
starting inference, also pass:

```text
--initialize-from-dataset
```

The initialization speed is independent from the policy action clamp. Its
default is `0.1 rad/s` per joint and can be reduced:

```text
--initialization-speed-rad-s=0.05
--initialization-timeout-s=90
```

After the initial pose is reached, the runner holds that pose and waits for an
explicit `s` input before policy control begins.

This option commands real hardware. Keep the workspace clear and have a
physical emergency stop available.

## Important Options

- `--start-frame`: frame offset inside the selected episode.
- `--actions-per-inference`: number of actions consumed before replanning.
- `--prefetch-threshold`: queue fraction that triggers background inference; default `0.5`.
- `--synchronous-inference`: disable prefetch and use the old blocking chunk behavior.
- `--frequency`: action consumption rate, normally 30 Hz for this dataset.
- `--max-action-delta-rad`: maximum target change per joint from measured state.
- `--initialization-speed-rad-s`: per-joint speed limit used only while moving to the initial pose.
- `--state-timeout-s`: stop dry-run when DDS arm state becomes stale.
- `--motion`: publish through `rt/arm_sdk`; otherwise use `rt/lowcmd`.
- `--root`: optional local LeRobot dataset root.

The runner currently accepts only ACT checkpoints with 28-dimensional state
and action schemas, and only controls the 14 arm joints of G1-29. It never
initializes a Dex3 controller.

For the 100-step ACT checkpoint on CPU, a practical starting point is:

```text
--actions-per-inference=75
--prefetch-threshold=0.5
```

This requests a new chunk when roughly 37 queued actions remain, providing
about 1.2 seconds of inference headroom at 30 Hz.
