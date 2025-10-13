from lerobot.datasets.lerobot_dataset import LeRobotDataset
import tqdm

episode_index = 1
dataset = LeRobotDataset(repo_id="unitreerobotics/G1_Dex3_ToastedBread_Dataset")

from_idx = dataset.meta.episodes["dataset_from_index"][episode_index]
to_idx = dataset.meta.episodes["dataset_to_index"][episode_index]
for step_idx in tqdm.tqdm(range(from_idx, to_idx)):
    step = dataset[step_idx]
