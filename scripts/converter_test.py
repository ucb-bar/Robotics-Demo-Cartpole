import numpy as np
import torch
import torch.nn as nn



class MLP(torch.nn.Module):
    def __init__(self, n_obs, n_acs):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(n_obs, 8)
        self.fc2 = torch.nn.Linear(8, 8)
        self.fc3 = torch.nn.Linear(8, n_acs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=-1)

        return x

model = MLP(4, 2)


# Load model
model.load_state_dict(torch.load("cartpole_agent_8.pt"))


def asBytes(activation: np.ndarray) -> bytes:
    flattened = activation.astype(np.float32).flatten()
    return flattened.tobytes()


# store model weights as binary file
model_structure = list(model.named_modules())

w1 = model.state_dict().get("fc1.weight").numpy().T
b1 = model.state_dict().get("fc1.bias").numpy()
w2 = model.state_dict().get("fc2.weight").numpy().T
b2 = model.state_dict().get("fc2.bias").numpy()
w3 = model.state_dict().get("fc3.weight").numpy().T
b3 = model.state_dict().get("fc3.bias").numpy()

print("w1:\n", w1.shape)
print("b1:\n", b1.shape)

with open("model.bin", "wb") as f:
    f.write(asBytes(w1))
    f.write(asBytes(b1))
    f.write(asBytes(w2))
    f.write(asBytes(b2))
    f.write(asBytes(w3))
    f.write(asBytes(b3))
    print("W1 size:", len(asBytes(w1)))
    print("B1 size:", len(asBytes(b1)))
    print("W2 size:", len(asBytes(w2)))
    print("B2 size:", len(asBytes(b2)))
    print("W3 size:", len(asBytes(w3)))
    print("B3 size:", len(asBytes(b3)))

# Test model
test_input = np.array([
    [0., 0., 0., 0.],
    ], dtype=np.float32)
test_tensor = torch.tensor(test_input, dtype=torch.float32)

output = model.forward(test_tensor)
print("model result:")
# [[0.5387, 0.4613]]
print(output)

print("raw result:")
fc1 = np.maximum(0, test_input @ w1 + b1)
fc2 = np.maximum(0, fc1 @ w2 + b2)
fc3 = fc2 @ w3 + b3
# fc3 = np.exp(fc3)
# fc3 /= fc3.sum()

print(fc3)


