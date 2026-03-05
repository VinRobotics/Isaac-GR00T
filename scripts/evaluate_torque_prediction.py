from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.experiment.data_config import load_data_config
from gr00t.eval.robot import RobotInferenceClient
import matplotlib.pyplot as plt
import numpy as np
data_config = load_data_config("vrh3_effort")
modality_config = data_config.modality_config()
modality_transform = data_config.transform()

pick_dataset = LeRobotSingleDataset(
    "/home/locht1/Documents/locht1/code_convert/output/20251204_VR_H3_pickpart_speedup1",
    embodiment_tag="new_embodiment",
    video_backend="torchvision_av",
    modality_configs=modality_config,
    # transforms=modality_transform
)

place_dataset = LeRobotSingleDataset(
    "/home/locht1/Documents/locht1/code_convert/output/20251206_VR_H3_placepart_speedup1",
    embodiment_tag="new_embodiment",
    video_backend="torchvision_av",
    modality_configs=modality_config,
    # transforms=modality_transform
)
ta_client = RobotInferenceClient()

ta_fast_client = RobotInferenceClient(port=6666)

total_ta_err = []
total_ta_fast_err = []
for i in range(0, 2000):
    data = place_dataset[i]
    ta_action = ta_client.get_action(data)
    ta_fast_action = ta_fast_client.get_action(data)

    effort = data["effort.left_arm"][16:]
    ta_pred_effort = ta_action["effort.left_arm"][0]
    ta_fast_pred_effort = ta_fast_action["effort.left_arm"][0]

    err = np.absolute(effort - ta_pred_effort).mean()
    ta_fast_err = np.absolute(effort - ta_fast_pred_effort).mean()

    # visualize line chart at each step
    for j in range(effort.shape[-1]):
        plt.figure()
        plt.plot(effort[:, j], label="Ground Truth")
        plt.plot(ta_pred_effort[:, j], label="Prediction")
        plt.plot(ta_fast_pred_effort[:, j], label="TA FAST Prediction")
        plt.legend(["Ground Truth", "Prediction", "TA FAST Prediction"])
        plt.title(f"Sample {i} Joint {j}, Err {np.round(err, 3)}, Err TA FAST {np.round(ta_fast_err, 3)}")
        plt.savefig(f"./eval_vs/torque_prediction_error_sample_{i}_joint_{j}.png")
        plt.close()
    total_ta_err.append(err)
    total_ta_fast_err.append(ta_fast_err)

print("Torque Prediction Error Evaluation Complete")
print("Torque Aware Error", np.mean(total_ta_err))
print("Torque Aware Fast Error", np.mean(total_ta_fast_err))