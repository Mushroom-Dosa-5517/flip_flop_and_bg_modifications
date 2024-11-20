import torch
import torch.nn as nn
import torch.nn.functional as F

class FF(nn.Module):
    def __init__(self, units):
        super(FF, self).__init__()
        self.units = units
        self.j_h = nn.Linear(self.units, self.units)
        self.j_x = nn.Linear(self.units, self.units)
        self.k_h = nn.Linear(self.units, self.units)
        self.k_x = nn.Linear(self.units, self.units)

    def forward(self, inputs, prev_output):
        # print(inputs.shape, prev_output.shape)
        j = torch.sigmoid(self.j_x(inputs) + self.j_h(prev_output))
        k = torch.sigmoid(self.k_x(inputs) + self.k_h(prev_output))
        output = j * (1 - prev_output) + (1 - k) * prev_output
        return output

