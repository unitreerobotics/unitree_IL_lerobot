"""Component-level smoke tests for unitree_lerobot.eval_robot.

Each script instantiates ONE base component (from robot_control/ or image_server/)
and verifies basic read / minimal write behavior. Safety first: every motion test
requires the user to press 's' to arm, uses a very small amplitude, and stops on 'q'.

Run, e.g.:

    python -m unitree_lerobot.eval_robot.tests.test_arm    --arm G1_29 --joint 0
    python -m unitree_lerobot.eval_robot.tests.test_ee     --ee  dex3  --amplitude 0.05
    python -m unitree_lerobot.eval_robot.tests.test_camera --image-host 192.168.123.164
    python -m unitree_lerobot.eval_robot.tests.test_mobile --base-type only_height
"""
