import torch
import torch.nn as nn


class PINN(nn.Module):
    """
    Physics-Informed Neural Network for AirfRANS.

    Inputs
    ------
    x
    y
    inlet_u
    inlet_v
    sdf
    normal_x
    normal_y

    Outputs
    -------
    u
    v
    p
    """

    def __init__(
        self,
        input_dim=7,
        output_dim=3,
        hidden_dim=128,
        num_hidden=8,
        activation="tanh",
    ):
        super().__init__()

        if activation.lower() == "tanh":
            act = nn.Tanh
        elif activation.lower() == "gelu":
            act = nn.GELU
        elif activation.lower() == "relu":
            act = nn.ReLU
        elif activation.lower() == "silu":
            act = nn.SiLU
        else:
            raise ValueError("Unsupported activation")

        layers = []

        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(act())

        for _ in range(num_hidden - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act())

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)

        self.initialize()

    def initialize(self):

        for m in self.modules():

            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):

        out = self.network(x)

        u = out[:, 0:1]
        v = out[:, 1:2]
        p = out[:, 2:3]

        return u, v, p


if __name__ == "__main__":

    model = PINN()

    x = torch.randn(64, 7)

    u, v, p = model(x)

    print(u.shape)
    print(v.shape)
    print(p.shape)