from __future__ import division
import numpy as np
import torch
import torch.nn as nn
import os
import math
from config import *


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size1, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, output_size)

    def forward(self, Y_p):
        out = self.fc1(Y_p)
        out = self.relu1(out)
        out = self.fc2(out)
        return out


if __name__ == '__main__':
    print("test model.py")