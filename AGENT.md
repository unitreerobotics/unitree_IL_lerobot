# Agent Notes: Hybrid G1 Arm Deployment

Last updated: 2026-06-08, Asia/Ho_Chi_Minh.

## Goal

Deploy the toasted-bread ACT policy to a real Unitree G1 while only controlling the 14 arm joints.

Temporary hardware limitation:

- No Dex3 hands.
- No live cameras.
- Real robot provides only 14 G1 arm joint states.

Hybrid inference design:

- Policy input still matches training schema: 28D state and 4 dataset camera streams.
- `observation.state[:14]` is replaced with live G1 arm state.
- `observation.state[14:28]` remains dataset Dex3 state.
- Policy outputs `chunk_size x 28`; only `[:, :14]` is used for G1 arms.
- Dex3 actions are ignored.

## Important Files

- `unitree_lerobot/eval_robot/hybrid_arm_infer.py`
  - Main online hybrid inference runner.
  - Defaults to dry-run unless `--send-actions` is present.
  - Uses async prefetch by default.
- `unitree_lerobot/eval_robot/hybrid_arm_utils.py`
  - Pure helpers: state composition, action slicing, clamp, timestep range.
- `unitree_lerobot/eval_robot/robot_control/arm_state_reader.py`
  - Read-only DDS state reader for dry-run.
- `unitree_lerobot/eval_robot/robot_control/robot_arm.py`
  - Modified `G1_29_ArmController`:
    - initializes `q_target` from current arm state;
    - supports `initialize_dds=False` so hybrid runner can initialize DDS once with the selected interface.
- `unitree_lerobot/eval_robot/HYBRID_ARM_INFERENCE.md`
  - Human-facing docs.
- `test/test_hybrid_arm_utils.py`
  - Unit tests.
- `Useme.md`
  - Practical commands for the user.

## Current Git State

The user committed and pushed:

```text
branch: son-deploy-hybrid
commit: c35d90a
message: commit them
remote: origin/son-deploy-hybrid
```

After that, `Useme.md` and `AGENT.md` were updated/created and may be uncommitted.

## Dataset And Policy

Dataset:

```text
unitreerobotics/G1_Dex3_ToastedBread_Dataset
total_episodes: 418
total_frames: 352022
fps: 30
episode 0: dataset indices 0..619, 620 frames, about 20.67s
```

State/action schema:

```text
28 dims = 14 G1 arm joints + 14 Dex3 joints
```

Camera keys:

```text
observation.images.cam_left_high
observation.images.cam_right_high
observation.images.cam_left_wrist
observation.images.cam_right_wrist
```

Best dataset video views for checking task behavior:

```text
cam_left_high
cam_right_high
```

Local checkpoint used:

```text
/home/jkl0909/code/Son/unitree_lerobot/unitree_lerobot/lerobot/outputs/g1_dex3_toastedbread_act/checkpoints/last/pretrained_model
```

Policy config:

```text
ACT
chunk_size: 100
n_action_steps: 100
temporal_ensemble_coeff: null
```

## Robot Mode Findings

`MotionSwitcher=ai` alone is not enough to prove the robot is in the correct sub-mode.

For `rt/arm_sdk`, the robot must be in Regular motion-control mode:

```text
R3 remote sequence:
L2 + B   -> Damping mode
L2 + UP  -> Locked Standing
R1 + X   -> Regular mode
```

Avoid:

```text
R2 + A   -> Running mode
L2 + R2  -> Debug/development mode
```

Debug/development mode is for `rt/lowcmd` full-body control and is risky here because the hybrid task only intends to control arms.

## Bugs Found And Fixed

1. Initial robot did not move with `rt/arm_sdk`.
   - Cause was not simply missing `--motion`; logs showed `motion=true`.
   - Key operational fix: put robot in Regular mode with `R1+X`.

2. DDS interface handling.
   - Hybrid runner initializes DDS with `ChannelFactoryInitialize(0, args.network_interface)`.
   - `G1_29_ArmController(..., initialize_dds=False)` prevents a second init without interface.

3. Unsafe initial target behavior.
   - Controller now sets `self.q_target` from current arm state during init, not zero.

4. Initialization ramp.
   - Added `--initialize-from-dataset`.
   - Added gradual motion controls:
     - `--initialization-speed-rad-s`
     - `--initialization-max-tracking-error-rad`
     - `--initialization-timeout-s`
   - Initialization stops and waits for user to enter `s` before policy starts.

5. CPU inference delay.
   - CPU takes about 0.95-1.05s per ACT chunk.
   - Old runner was synchronous:
     - infer chunk;
     - execute chunk;
     - infer next chunk.
   - New runner uses local async prefetch:
     - main thread publishes queued actions at 30 Hz;
     - one background worker infers next chunk;
     - prefetch starts when queue falls below `--prefetch-threshold`.

## Async Prefetch Details

Default parameters now:

```text
--actions-per-inference=75
--prefetch-threshold=0.5
```

At 30 Hz, 75 actions last 2.5s. Prefetch at 50% gives about 1.25s inference headroom, enough for current CPU.

Action queue behavior:

- Actions are keyed by absolute dataset timestep.
- New chunk has an anchor timestep.
- Stale actions are discarded.
- Overlapping future actions are replaced by the newer chunk.
- If queue is empty and inference is not ready, the loop waits and logs `queue_wait_s`.

CSV new fields:

```text
inference_anchor
action_queue_size
queue_wait_s
```

## Recommended Commands

See `Useme.md`. Main real-control test:

```bash
python unitree_lerobot/eval_robot/hybrid_arm_infer.py \
  --policy-path=/home/jkl0909/code/Son/unitree_lerobot/unitree_lerobot/lerobot/outputs/g1_dex3_toastedbread_act/checkpoints/last/pretrained_model \
  --episode=0 \
  --max-policy-steps=120 \
  --actions-per-inference=75 \
  --prefetch-threshold=0.5 \
  --frequency=30 \
  --initialization-speed-rad-s=0.05 \
  --initialization-max-tracking-error-rad=0.05 \
  --initialization-timeout-s=120 \
  --max-action-delta-rad=0.02 \
  --network-interface=enp1s0 \
  --motion \
  --initialize-from-dataset \
  --send-actions \
  --control-confirmation=SEND_TO_REAL_G1
```

Ramp-up sequence:

```text
120 steps -> 300 steps -> full episode 0, 620 steps
```

## Verification Already Done

Commands that passed:

```bash
/home/jkl0909/.holosoma_deps/miniconda3/envs/unitree_lerobot/bin/python -m compileall -q unitree_lerobot/eval_robot/hybrid_arm_infer.py unitree_lerobot/eval_robot/hybrid_arm_utils.py unitree_lerobot/eval_robot/robot_control/robot_arm.py
/home/jkl0909/.holosoma_deps/miniconda3/envs/unitree_lerobot/bin/python test/test_hybrid_arm_utils.py
git diff --check
```

Unit test result:

```text
Ran 5 tests
OK
```

Simulated 300 timestep queue schedule:

```text
submissions [37, 74, 111, 148, 185, 222, 259, 296]
underruns []
```

Dry-run DDS was attempted after async changes but failed because `enp1s0` was `DOWN/NO-CARRIER`, before inference/control:

```text
enp1s0: does not match an available interface
channel factory init error
```

This was an environment/network issue, not a code failure.

## Observed Runtime Behavior Before Async

Real robot reached dataset initial arm pose after switching to Regular mode.

`--max-policy-steps=100 --actions-per-inference=100` executed 100 rows/actions, but motion looked small because:

- first 100 timesteps are only about 3.33s;
- encoder max change was about 0.118 rad;
- then code stopped due to max-policy-steps.

Full episode 0 is 620 timesteps.

## Safety Notes

- `dry_run=False` means real commands are being sent.
- Do not enter `s` after initialization unless arms are stable and workspace is clear.
- Keep emergency stop ready.
- Do not use Debug mode for this hybrid runner.
- Do not increase `frequency` above 30; dataset and policy are trained at 30 Hz.
- `--max-action-delta-rad=0.02` is the current recommended clamp.
  - It corresponds to about 0.6 rad/s at 30 Hz.
  - Increase only after reviewing CSV and observing safe behavior.
