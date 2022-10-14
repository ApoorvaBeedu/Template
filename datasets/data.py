# Copyright (c) Facebook, Inc. and its affiliates.

import os

import torch
import torchvision
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DistributedSampler
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

import utils.transforms as T
from utils.pytorch_utils import Transform

__all__ = [
    "get_dataset",
]


def get_transform(args):
    """Creates custom transforms for data loading and augmentations

    Args:
        args (_type_): _description_

    Returns:
        returb dataloaders
    """
    transform_train = torchvision.transforms.transforms.Compose()
    transform_eval = torchvision.transforms.transforms.Compose()
    return transform_train, transform_eval


def get_dataset(args):
    """Gets the dataset for the dataloader

    Args:
        args (_type_): _description_

    Returns:
        Returns dataloaders
    """

    # Edit below to suit your needs
    from datasets.dataloader import Dataloader
    transforms_train, transforms_eval = get_transform(args, compose=True)
    train_dset = Dataloader(**args)
    val_dset = Dataloader(**args)

    train_dataloader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=args.batch_size // args.world_size,
        sampler=DistributedSampler(train_dset),
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        multiprocessing_context='fork')

    val_dataloader = torch.utils.data.DataLoader(
        val_dset,
        batch_size=args.batch_size // args.world_size,
        sampler=DistributedSampler(val_dset, shuffle=False),
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        multiprocessing_context='fork')

    return {"train": train_dataloader, "test": val_dataloader}
