from enum import Enum
from typing import Dict, TypedDict

import torch
from utils import pytorch_utils as util

float_pt = torch.FloatTensor


class TrainingSample(TypedDict):
    images: torch.Tensor
    filename: torch.Tensor


class Output(TypedDict):
    output: torch.Tensor


class LossDict():

    def __init__(self):
        self.loss_avg = util.RunningAverage()

    def __call__(self):
        return self.loss_avg()

    def get_loss_avg(self):
        return self.__call__()


class Losses(TypedDict):
    train: LossDict = LossDict()
    test: LossDict = LossDict()


class Split(Enum):
    Train = "train"
    Test = "test"
    Evaluate = "eval"
