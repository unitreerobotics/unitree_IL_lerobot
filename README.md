<div align="center">
  <h1 align="center"> unitree_lerobot </h1>
  <h3 align="center"> Unitree Robotics </h3>
  <p align="center">
    <a href="./README.md"> English </a> | <a href="./docs/README_zh.md">中文</a>
  </p>
    <p align="center">
     <a href="https://discord.gg/ZwcVwxv5rq" target="_blank"><img src="https://img.shields.io/badge/-Discord-5865F2?style=flat&logo=Discord&logoColor=white" alt="Unitree LOGO"></a>
  </p>
</div>

| Unitree Robotics repositories                      | link                                                                               |
| -------------------------------------------------- | ---------------------------------------------------------------------------------- |
| Unitree Datasets                                   | [unitree datasets](https://huggingface.co/unitreerobotics)                         |
| AVP Teleoperate                                    | [avp_teleoperate](https://github.com/unitreerobotics/avp_teleoperate)              |
| Unitree Sim IsaacLab                               | [unitree_sim_isaaclab](https://github.com/unitreerobotics/unitree_sim_isaaclab)    |
| Conversion of various versions of lerobot datasets | [any4lerobot](https://github.com/Tavish9/any4lerobot/tree/main/ds_version_convert) |

# 🔖 Release Note

### 🏷️ v0.3

1.Update [`lerobot dataset v3.0`](https://github.com/huggingface/lerobot/blob/main/docs/source/porting_datasets_v3.mdx).

2.More policy support([`pi05`](https://github.com/huggingface/lerobot/tree/main/src/lerobot/policies/pi05), [`groot`](https://github.com/huggingface/lerobot/tree/main/src/lerobot/policies/groot)).

### 🏷️ v0.2

1.Add `data conversion` and `model deployment` for `brainco` and `inspire1` Dexterous hands.

2.Add the functionality of `replaying the robot dataset`.

3.Add `simulation environment verification` [unitree_sim_isaaclab].

### 🏷️ v0.1

Support `data conversion`, `model deployment`, and `real-world testing` for `G1 + Dex1 + Dex3`.

# 0. 📖 Introduction

This repository is used for `lerobot training validation`(Supports LeRobot datasets version 2.0 and above.) and `unitree data conversion`.

`❗Tips： If you have any questions, ideas or suggestions that you want to realize, please feel free to raise them at any time. We will do our best to solve and implement them.`

| Directory  | Description                                                                                              |
| ---------- | -------------------------------------------------------------------------------------------------------- |
| lerobot    | The code in the `lerobot repository` for training; its corresponding commit version number is `0878c68`. |
| utils      | `unitree data processing tool `                                                                          |
| eval_robot | `unitree real machine inference verification of the model`                                               |

# 1. 📦 Environment Setup

## 1.1 🦾 LeRobot Environment Setup

The purpose of this project is to use the [LeRobot](https://github.com/huggingface/lerobot) open-source framework to train and test data collected from Unitree robots. Therefore, it is necessary to install the LeRobot-related dependencies first. The installation steps are as follows, and you can also refer to the official [LeRobot](https://github.com/huggingface/lerobot) installation guide:

```bash
# Clone the source code
git clone --recurse-submodules https://github.com/unitreerobotics/unitree_lerobot.git

# If already downloaded:
git submodule update --init --recursive

# Create a conda environment
conda create -y -n unitree_lerobot python=3.10
conda activate unitree_lerobot
conda install pinocchio -c conda-forge

conda install ffmpeg=7.1.1 -c conda-forge

# Install LeRobot
cd unitree_lerobot/lerobot && pip install -e .

# Install unitree_lerobot
cd ../../ && pip install -e .
```

## 1.2 🕹️ unitree_sdk2_python

For `DDS communication` on Unitree robots, some dependencies need to be installed. Follow the installation steps below:

```bash
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python  && pip install -e .
```

# 2. ⚙️ Data Collection and Conversion

## 2.1 🖼️ Load Datasets

If you want to directly load the dataset we have already recorded,
Load the [`unitreerobotics/G1_Dex3_ToastedBread_Dataset`](https://huggingface.co/datasets/unitreerobotics/G1_Dex3_ToastedBread_Dataset) dataset from Hugging Face. The default download location is `~/.cache/huggingface/lerobot/unitreerobotics`. If you want to load data from a local source, please change the `root` parameter.

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import tqdm

episode_index = 1
dataset = LeRobotDataset(repo_id="unitreerobotics/G1_Dex3_ToastedBread_Dataset")

from_idx = dataset.meta.episodes["dataset_from_index"][episode_index]
to_idx = dataset.meta.episodes["dataset_to_index"][episode_index]

for step_idx in tqdm.tqdm(range(from_idx, to_idx)):
    step = dataset[step_idx]
```

`visualization`

```bash
cd unitree_lerobot/lerobot

python src/lerobot/scripts/lerobot_dataset_viz.py \
    --repo-id unitreerobotics/G1_Dex3_ToastedBread_Dataset \
    --episode-index 0
```

## 2.2 🔨 Data Collection

If you want to record your own dataset. The open-source teleoperation project [avp_teleoperate](https://github.com/unitreerobotics/avp_teleoperate/tree/g1) can be used to collect data using the Unitree G1 humanoid robot. For more details, please refer to the [avp_teleoperate](https://github.com/unitreerobotics/avp_teleoperate/tree/g1) project.

## 2.3 ✂️Data Processing

When you finished the Data Collection step, it may be essential for you to process the data you have been collected. For example, if one of the episodes failed to complete the task, you need to delete the episode, to increase the data's quality.

We have designed a data editor in this project, in order to help users process the data. Currently, it can cut off excess segments in the episodes, and delete the bad episodes.

![](https://oss-global-cdn.unitree.com/static/986a520f8b444784b090a07b9efc6803_1455x1071.jpg)

Before you launch the data editor, you need to install pyqt5.

```bash
conda activate unitree_lerobot
pip install PyQt5
```

Launch the data editor as below.

```bash
cd data_editor
python data_editor_EN.py
```

At first, click the button `Select Dataset Path`, choose the path of your dataset. For example, at the path tree below, `test_dataset/`is the path you need to choose.

```bash
test_dataset/
    ├── episode_0001
    ├── episode_0003
    ├── episode_0004
    ├── ...
```

You can drag the red line directly to seek playback, and drag the red line with Shift on your keyboard to select the playback range, and then trim the selected range with the button `Trim Selected Range`. If you want to delete the current episode, click the button `Delete Current Episode`.

## 2.4 🛠️ Data Conversion

The data collected using [avp_teleoperate](https://github.com/unitreerobotics/avp_teleoperate/tree/g1) is stored in JSON format. Assuming the collected data is stored in the `$HOME/datasets/task_name`, the format is as follows

```
datasets/                               # Dataset folder
    └── task_name /                     # Task name
        ├── episode_0001                # First trajectory
        │    ├──audios/                 # Audio information
        │    ├──colors/                 # Image information
        │    ├──depths/                 # Depth image information
        │    └──data.json               # State and action information
        ├── episode_0002
        ├── episode_...
        ├── episode_xxx
```

### 2.4.1 🔀 Sort and Rename

When generating datasets for LeRobot, it is recommended to ensure that the data naming convention, starting from `episode_0`, is sequential and continuous. You can use the following script to `sort and rename` the data accordingly.

```bash
python unitree_lerobot/utils/sort_and_rename_folders.py \
        --data_dir $HOME/datasets/task_name
```

#### 2.4.2 🔄 Conversion

Convert `Unitree JSON` Dataset to `LeRobot` Format. You can define your own `robot_type` based on [ROBOT_CONFIGS](https://github.com/unitreerobotics/unitree_lerobot/blob/main/unitree_lerobot/utils/convert_unitree_json_to_lerobot.py#L154).

```bash
# --raw-dir     Corresponds to the directory of your JSON dataset
# --repo-id     Your unique repo ID on Hugging Face Hub
# --push_to_hub Whether or not to upload the dataset to Hugging Face Hub (true or false)
# --robot_type  The type of the robot used in the dataset (e.g., Unitree_Z1_Single, Unitree_Z1_Dual, Unitree_G1_Dex1, Unitree_G1_Dex3, Unitree_G1_Brainco, Unitree_G1_Inspire,Unitree_G1_Dex1_Sim)

python unitree_lerobot/utils/convert_unitree_json_to_lerobot.py \
    --raw-dir $HOME/datasets \
    --repo-id your_name/repo_task_name \
    --robot_type Unitree_G1_Dex3 \
    --push_to_hub
```

**Node:** `Unitree_G1_Dex1_Sim` is a robot type used for data collection in unitree_sim_isaaclab
, with the head equipped with a single-viewpoint camera.

# 3. 🚀 Training

[For training, please refer to the official LeRobot training example and parameters for further guidance.](https://github.com/huggingface/lerobot/tree/main/docs/source)

- `Train Act Policy` [Please refer to it in detail](https://github.com/huggingface/lerobot/blob/main/docs/source/act.mdx)

```bash
cd unitree_lerobot/lerobot

python src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=unitreerobotics/G1_Dex3_ToastedBread_Dataset \
    --policy.push_to_hub=false \
    --policy.type=act
```

- `Train Diffusion Policy` [Please refer to it in detail](https://github.com/huggingface/lerobot/blob/main/docs/source/policy_diffusion_README.md)

```bash
cd unitree_lerobot/lerobot

python src/lerobot/scripts/lerobot_train.py\
    --dataset.repo_id=unitreerobotics/G1_Dex3_ToastedBread_Dataset \
    --policy.push_to_hub=false \
    --policy.type=diffusion
```

- `Train Pi0 Policy` [Please refer to it in detail](https://github.com/huggingface/lerobot/blob/main/docs/source/pi0.mdx)

```bash
cd unitree_lerobot/lerobot

python src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=unitreerobotics/G1_Dex3_ToastedBread_Dataset \
    --policy.push_to_hub=false \
    --policy.type=pi0
```

- `Train Pi05 Policy` [Please refer to it in detail](https://github.com/huggingface/lerobot/blob/main/docs/source/pi05.mdx)

```bash
cd unitree_lerobot/lerobot

python src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=unitreerobotics/G1_Dex3_ToastedBread_Dataset \
    --policy.type=pi05 \
    --output_dir=./outputs/pi05_training \
    --job_name=pi05_training \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --policy.push_to_hub=false
```

- `Train Gr00t Policy` [Please refer to it in detail](https://github.com/huggingface/lerobot/blob/main/docs/source/groot.mdx)

```bash
cd unitree_lerobot/lerobot

python src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=unitreerobotics/G1_Dex3_ToastedBread_Dataset \
    --output_dir=./outputs/groot_training \
    --policy.push_to_hub=false \
    --policy.type=groot \
    --policy.tune_diffusion_model=false \
    --job_name=groot_training
```

If you want to use multi-GPU training, please refer to the details [here](https://github.com/huggingface/lerobot/blob/main/docs/source/multi_gpu_training.mdx)

# 4. 🤖 Real-World Testing

To test your trained model on a real robot, you can use the eval_g1.py script located in the eval_robot folder. Here’s how to run it:

[To open the image_server, follow these steps](https://github.com/unitreerobotics/avp_teleoperate?tab=readme-ov-file#31-%EF%B8%8F-image-server)

```bash

# --policy.path: Specifies the path to the pre-trained model, used for evaluating the policy.
# --repo_id: The repository ID of the dataset, used to load the dataset required for evaluation.
# --root: The root directory path of the dataset, defaults to an empty string.
# --episodes: The number of evaluation episodes; setting it to 0 uses the default value.
# --frequency: The evaluation frequency (in Hz), used to control the time step of the evaluation.
# --arm: The model of the robotic arm, (e.g., G1_29, G1_23).
# --ee: The type of end-effector, (e.g., dex3, dex1, inspire1, brainco).
# --visualization: Whether to enable visualization; setting it to true enables it.
# --send_real_robot: Whether to send commands to the real robot.


python unitree_lerobot/eval_robot/eval_g1.py  \
    --policy.path=unitree_lerobot/lerobot/outputs/train/2025-03-25/22-11-16_diffusion/checkpoints/100000/pretrained_model \
    --repo_id=unitreerobotics/G1_Dex3_ToastedBread_Dataset \
    --root="" \
    --episodes=0 \
    --frequency=30 \
    --arm="G1_29" \
    --ee="dex3" \
    --visualization=true \

If you want to run inference tests in the unitree_sim_isaaclab simulation environment, please execute:

# --save_data: Allows recording data while running inference. At present, this option is limited to the sim environment.
# --task_dir: the directory where data is stored
# --max_episodes: the maximum number of inference runs per task; if exceeded, the task is considered failed by default

python unitree_lerobot/eval_robot/eval_g1_sim.py  \
    --policy.path=unitree_lerobot/lerobot/outputs/train/2025-03-25/22-11-16_diffusion/checkpoints/100000/pretrained_model \
    --repo_id=unitreerobotics/G1_Dex3_ToastedBread_Dataset \
    --root="" \
    --episodes=0 \
    --frequency=30 \
    --arm="G1_29" \
    --ee="dex3" \
    --visualization=true \
    --save_data=false \
    --task_dir="./data" \
    --max_episodes=1200

# If you want to evaluate the model's performance on the dataset, use the command below for testing
python unitree_lerobot/eval_robot/eval_g1_dataset.py  \
    --policy.path=unitree_lerobot/lerobot/outputs/train/2025-03-25/22-11-16_diffusion/checkpoints/100000/pretrained_model \
    --repo_id=unitreerobotics/G1_Dex3_ToastedBread_Dataset \
    --root="" \
    --episodes=0 \
    --frequency=30 \
    --arm="G1_29" \
    --ee="dex3" \
    --visualization=true \
    --send_real_robot=false
```

**Note:** If you are using the `unitree_sim_isaaclab` simulation environment, please refer to [unitree_sim_isaaclab](https://github.com/unitreerobotics/unitree_sim_isaaclab) for environment setup and usage instructions.

# 5. 🎬 Replay Datasets On Robot

This section provides instructions on how to replay datasets on the robot.
It is useful for testing and validating the robot's behavior using pre-recorded data.

```bash

# --repo_id         Dataset repository ID on Hugging Face Hub (e.g., unitreerobotics/G1_Dex3_ToastedBread_Dataset)
# --root            Path to the root directory of the dataset (leave empty to use the default cache path)
# --episodes        Index of the episode to replay (e.g., 0 for the first episode)
# --frequency       Replay frequency in Hz (e.g., 30 for 30 frames per second)
# --arm             Type of robot arm used (e.g., G1_29, G1_23)
# --ee              Type of end-effector used (e.g., dex3, dex1, inspire1, brainco)
# --visualization   Enable or disable visualization during replay (true for enabling, false for disabling)

python unitree_lerobot/eval_robot/replay_robot.py \
    --repo_id=unitreerobotics/G1_Dex3_ToastedBread_Dataset \
    --root="" \
    --episodes=0 \
    --frequency=30 \
    --arm="G1_29" \
    --ee="dex3" \
    --visualization=true
```

# 6. 🤔 Troubleshooting

| Problem                                                                                                                                                                                                                                     | Solution                                                       |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| **Why use `LeRobot v2.0`?**                                                                                                                                                                                                                 | [Explanation](https://github.com/huggingface/lerobot/pull/461) |
| **401 Client Error: Unauthorized** (`huggingface_hub.errors.HfHubHTTPError`)                                                                                                                                                                | Run `huggingface-cli login` to authenticate.                   |
| **FFmpeg-related errors:** <br> Q1: `Unknown encoder 'libsvtav1'` <br> Q2: `FileNotFoundError: No such file or directory: 'ffmpeg'` <br> Q3: `RuntimeError: Could not load libtorchcodec. Likely causes: FFmpeg is not properly installed.` | Install FFmpeg: <br> `conda install -c conda-forge ffmpeg`     |
| **Access to model `google/paligemma-3b-pt-224` is restricted.**                                                                                                                                                                             | Run `huggingface-cli login` and request access if needed.      |

# 7. 🙏 Acknowledgement

This code builds upon following open-source code-bases. Please visit the URLs to see the respective LICENSES (If you find these projects valuable, it would be greatly appreciated if you could give them a star rating.):

1. https://github.com/huggingface/lerobot
2. https://github.com/unitreerobotics/unitree_sdk2_python
3. https://github.com/unitreerobotics/xr_teleoperate
4. https://github.com/unitreerobotics/unitree_sim_isaaclab
