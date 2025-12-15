import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def plot_trajectory(
    info,
    save_plot_path=None,
):
    """Simple plot of the trajectory with state, gt action, and pred action."""

    # Use non interactive backend for matplotlib if headless
    if save_plot_path is not None:
        matplotlib.use("Agg")

    action_dim = info["action_dim"]
    prev_pred_action_across_time = info["prev_pred_action_across_time"]
    pred_action_across_time = info["pred_action_across_time"]

    if prev_pred_action_across_time is None:
        return

    # Adjust figure size and spacing to accommodate titles
    fig, axes = plt.subplots(nrows=action_dim, ncols=1, figsize=(10, 4 * action_dim + 2))

    # Leave plenty of space at the top for titles
    plt.subplots_adjust(top=0.92, left=0.1, right=0.96, hspace=0.4)

    print("Creating visualization...")

    # Combine all modality keys into a single string
    # add new line if total length is more than 60 chars
    executed_horizon = 5

    x = np.arange(0,pred_action_across_time.shape[1] + executed_horizon)

    # Loop through each action dim
    for i, ax in enumerate(axes):
        # The dimensions of state_joints and action are the same only when the robot uses actions directly as joint commands.
        # Therefore, do not plot them if this is not the case.
        # if state_joints_across_time.shape == gt_action_across_time.shape:
        #     ax.plot(state_joints_across_time[:, i], label="state joints", alpha=0.7)
        ax.plot(prev_pred_action_across_time[0][:, i].cpu(), label="prev", linewidth=2)
        ax.plot(x[executed_horizon:], pred_action_across_time[0][:, i].cpu(), label="current", linewidth=2)

        ax.set_title(f"Action Dimension {i}", fontsize=12, fontweight="bold", pad=10)
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(True, alpha=0.3)

        # Set better axis labels
        ax.set_xlabel("Time Step", fontsize=10)
        ax.set_ylabel("Value", fontsize=10)

    if save_plot_path:
        print("saving plot to", save_plot_path)
        plt.savefig(save_plot_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()