import numpy as np
import torch
from pathlib import Path
from run_cosim import MNISTModel

# Absolute path to model_weights.npz
npz_path = Path(__file__).parent.parent / "model" / "model_weights.npz"

print("Loading:", npz_path)

data = np.load(npz_path)
data = dict(data)

model = MNISTModel()
state_dict = model.state_dict()

for k in state_dict.keys():
    np_key = k.replace('.', '_')
    if np_key in data:
        state_dict[k] = torch.tensor(data[np_key])

torch.save(state_dict, Path(__file__).parent.parent / "model" / "model_weights.pth")
print("Converted → model_weights.pth")
