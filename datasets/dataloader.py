# Copyright (c) Facebook, Inc. and its affiliates.
"""The Epic Kitchens dataset loaders."""

import csv
import logging
import pickle as pkl
from collections import OrderedDict
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class Dataloader(Dataset):
    """dataloader."""

    def __init__(
        self,
        **other_kwargs,
    ):
        """
        Args:
            action_labels_fpath (Path): Path to map the verb and noun labels to
                actions. It was used in the anticipation paper, that defines
                a set of actions and train for action prediction, as opposed
                to verb and noun prediction.
            annotation_dir: Where all the other annotations are typically stored
        """
        self.df = None # DataFrame for the training annotations
       
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        pass
