import sys
import os
import torch
import tkinter as tk
from tkinter import filedialog
from collections import OrderedDict
from rlgym_ppo.ppo import DiscreteFF, ValueEstimator


def model_info_from_dict(loaded_dict):
    state_dict = OrderedDict(loaded_dict)
    bias_counts, weight_counts = [], []
    for key, value in state_dict.items():
        if ".weight" in key:
            weight_counts.append(value.numel())
        if ".bias" in key:
            bias_counts.append(value.size(0))
    inputs = int(weight_counts[0] / bias_counts[0])
    outputs = bias_counts[-1]
    layer_sizes = bias_counts[:-1]
    return inputs, outputs, layer_sizes


def rename_model_state_dict(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if not key.startswith("model."):
            new_state_dict["model." + key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def make_models_from_dicts(policy_state_dict, critic_state_dict):
    policy_inputs, policy_outputs, policy_sizes = model_info_from_dict(policy_state_dict)
    critic_inputs, critic_outputs, critic_sizes = model_info_from_dict(critic_state_dict)
    device = torch.device("cpu")
    policy = DiscreteFF(policy_inputs, policy_outputs, policy_sizes, device)
    critic = ValueEstimator(critic_inputs, critic_sizes, device)
    return policy, critic


def main():
    # Ask which conversion direction to use
    mode = None
    while mode not in ("to_cpp", "to_python"):
        mode = input("Enter mode ('to_cpp' or 'to_python'): ").strip().lower()

    # GUI folder picker
    root = tk.Tk()
    root.withdraw()
    print("Select the folder containing your checkpoint files...")
    path = filedialog.askdirectory(title="Select checkpoint folder")
    root.destroy()

    if not path:
        sys.exit("No folder selected. Exiting.")

    print(f"Selected folder: {path}")

    # Output folder next to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "CPP_CHECKPOINT" if mode == "to_cpp" else "PYTHON_CHECKPOINT")
    os.makedirs(output_path, exist_ok=True)

    if mode == "to_cpp":
        print("Converting Python → C++ TorchScript (.LT)")

        policy_state_dict = torch.load(os.path.join(path, "PPO_POLICY.pt"))
        critic_state_dict = torch.load(os.path.join(path, "PPO_VALUE_NET.pt"))

        policy, critic = make_models_from_dicts(policy_state_dict, critic_state_dict)
        policy.load_state_dict(policy_state_dict)
        critic.load_state_dict(critic_state_dict)

        torch.jit.save(torch.jit.script(policy.model), os.path.join(output_path, "POLICY.LT"))
        torch.jit.save(torch.jit.script(critic.model), os.path.join(output_path, "CRITIC.LT"))

        print(f"✅ TorchScript models saved to: {output_path}")

    else:
        print("Converting C++ TorchScript (.LT) → Python checkpoint (.PT)")

        policy_path = os.path.join(path, "POLICY.LT")
        critic_path = os.path.join(path, "CRITIC.LT")

        if not (os.path.exists(policy_path) and os.path.exists(critic_path)):
            sys.exit("❌ Missing 'POLICY.LT' or 'CRITIC.LT' files in the selected folder.")

        policy_ts = torch.jit.load(policy_path, map_location="cpu")
        critic_ts = torch.jit.load(critic_path, map_location="cpu")

        policy_py, critic_py = make_models_from_dicts(policy_ts.state_dict(), critic_ts.state_dict())
        policy_optim = torch.optim.Adam(policy_py.parameters())
        critic_optim = torch.optim.Adam(critic_py.parameters())

        torch.save(rename_model_state_dict(policy_ts.state_dict()), os.path.join(output_path, "PPO_POLICY.pt"))
        torch.save(rename_model_state_dict(critic_ts.state_dict()), os.path.join(output_path, "PPO_VALUE_NET.pt"))
        torch.save(policy_optim.state_dict(), os.path.join(output_path, "PPO_POLICY_OPTIMIZER.pt"))
        torch.save(critic_optim.state_dict(), os.path.join(output_path, "PPO_VALUE_NET_OPTIMIZER.pt"))

        print(f"✅ Python checkpoints saved to: {output_path}")

    print("\nDone! 🚀")
    print(f"Output folder: {output_path}")


if __name__ == "__main__":
    main()
