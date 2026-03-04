# %%
import numpy as np
from transformers import AutoProcessor

# Load the tokenizer from the Hugging Face hub
tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)

# Tokenize & decode action chunks (we use dummy data here)
action_data = np.random.rand(256, 50, 14)    # one batch of action chunks
tokens = tokenizer(action_data)              # tokens = list[int]
decoded_actions = tokenizer.decode(tokens)


# %%
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.experiment.data_config import load_data_config

data_config = load_data_config("vrh3_effort")
modality_config = data_config.modality_config()
modality_transform = data_config.transform()

pick_dataset = LeRobotSingleDataset(
    "/home/locht1/Documents/locht1/code_convert/output/20251204_VR_H3_pickpart_speedup1",
    embodiment_tag="new_embodiment",
    video_backend="torchvision_av",
    modality_configs=modality_config,
    transforms=modality_transform
)

place_dataset = LeRobotSingleDataset(
    "/home/locht1/Documents/locht1/code_convert/output/20251206_VR_H3_placepart_speedup1",
    embodiment_tag="new_embodiment",
    video_backend="torchvision_av",
    modality_configs=modality_config,
    transforms=modality_transform
)

# %%
# from tqdm import tqdm
train_token_data = (
    [d["effort_history"] for d in pick_dataset] +
    [d["effort_history"] for d in place_dataset]
)
# %%
torque_data = np.asarray(train_token_data)

np.save("./fast_torque.npy", torque_data)

# %%
tokenizer = tokenizer.fit(torque_data)
tokenizer.save_pretrained("./fast_torque")
tokenizer.push_to_hub("locht131/fast_torque_tokenizer")


