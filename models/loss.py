import os.path as osp
from os import stat

import torch
import torch.nn.functional as F
from torch import nn

from models.enums import float_pt

# import pdb


class Loss(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.cross_entropy = nn.CrossEntropyLoss(weight=None,
                                                 reduction="mean")
        self.loss_l2 = torch.nn.MSELoss(reduction="none")
        self.loss_l2_rt = torch.nn.MSELoss(reduction="none")
        self.loss_l1 = torch.nn.L1Loss(reduction="none")
        self.loss_l1_depth = torch.nn.L1Loss(reduction="none")
        self.cosine = torch.nn.CosineSimilarity()

    def forward(self, est, gt):
        """_summary_

        Args:
            est (_type_): Estimated output
            gt (_type_): Ground truth labels

        Returns:
            loss values, can change it to be whatever
        """
        loss_ = torch.zeros(1).type_as(est[0])


        return loss_
