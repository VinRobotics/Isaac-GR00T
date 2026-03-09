from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.experiment.data_config import load_data_config
from gr00t.eval.robot import RobotInferenceClient, RobotInferenceServer
import numpy as np
import torch
from copy import deepcopy
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
import os
from PIL import Image

data_config = load_data_config("vrh3_two_hand_task_completion")
modality_config = data_config.modality_config()
modality_transform = data_config.transform()

dataset = LeRobotSingleDataset(
    "/home/locht1/Documents/locht1/code_convert/output/20251213_VR_H3_pickpart_speedup1_done_completion_test",
    embodiment_tag="new_embodiment",
    video_backend="torchvision_av",
    modality_configs=modality_config,
)
test_dataloader = DataLoader(dataset, batch_size=32)

# Create test folder for saving images
test_output_dir = "test_pick"
os.makedirs(test_output_dir, exist_ok=True)

client = RobotInferenceClient()
results = 0
i=0
for data in tqdm(dataset):
    obs = data
    # print(obs)
    action = client.get_action(obs)
    task_completion = action["task_completion"] >= 0.5
    
    correct = task_completion == obs["observation.tasks.done"][0][0]
    results += correct
    
    # Save image with prediction and label
    pred = int(task_completion)
    label = int(obs["observation.tasks.done"][0][0])
    
    # Get the image from observation (video.cam_front)
    img_tensor = obs["video.cam_front"][0]  # Get first frame
    if isinstance(img_tensor, torch.Tensor):
        img_array = img_tensor.cpu().numpy()
    else:
        img_array = np.array(img_tensor)
    
    # Handle different image formats (C, H, W) or (H, W, C)
    if img_array.shape[0] == 3:  # (C, H, W) format
        img_array = np.transpose(img_array, (1, 2, 0))
    
    # Normalize if needed (if values are in [0, 1] range)
    if img_array.max() <= 1.0:
        img_array = (img_array * 255).astype(np.uint8)
    else:
        img_array = img_array.astype(np.uint8)
    
    # Save image
    img = Image.fromarray(img_array)
    img_filename = f"img_{i}_pred_{pred}_label_{label}.png"
    img.save(os.path.join(test_output_dir, img_filename))
    
    i+=1
    print("current result", results/i)
    print(results)
print("final", results/len(dataset))
