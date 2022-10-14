import torch.nn as nn

from . import *
from .trainer import *


def weight_initialise(model):
    # Initialise with Kaiming normal
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal(m.weight,
                                   mode='fan_out',
                                   nonlinearity='relu')
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal(m.weight,
                                   mode='fan_out',
                                   nonlinearity='relu')
