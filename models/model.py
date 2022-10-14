import copy
import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision
from loguru import logger


class Model(nn.Module):

    def __init__(self, args, dataset):
        super().__init__()
        """Define the layers for the model here
        """
        self.args = args

    def get_params(self):
        params = [
            {
                "params": self.model.parameters(),
                "lr": self.args.lr
            },
        ]
        return params

    def forward(self, **kwargs):
        pass
