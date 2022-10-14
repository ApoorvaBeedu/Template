import json
import logging
import math
import os
import pdb
import random
import shutil
import warnings

import torch
from loguru import logger
from omegaconf import OmegaConf

# Ignore warnings
warnings.filterwarnings("ignore")


def reduce_dict(input_dict, world_size):
    """
    Args:
        input_dict (dict): all the values will be reduced
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    if world_size < 2:
        return input_dict
    with torch.inference_mode():
        names = []
        values = []
        for k in input_dict.keys():
            names.append(k)
            if type(input_dict[k]) is dict:
                values.append(reduce_dict(input_dict[k], world_size))
            else:
                torch.distributed.all_reduce(input_dict[k])
                values.append(input_dict[k] / world_size)
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def restore_from_file(args, model, optimizer, lr_scheduler, rank=0):
    """
    Loads weights saved model, assumes the checkpoint is saved as last_checkpoint_{}.tar.
    Takes in complete path.
    :param args: Args
    :param model: Model to load to
    :return:
    """
    if os.path.isfile(args.restore_file):
        checkpoint_path = args.restore_file
    else:
        if os.path.isabs(args.restore_file):
            restore_path = args.restore_file
        else:
            restore_path = os.path.join(args.model_dir_path, args.restore_file)

        counter = int(
            sorted(os.listdir(restore_path))[-1].split("_")[-1].split(".")[0])
        if args.restore_best:
            if os.path.exists(os.path.join(restore_path, "checkpoint.txt")):
                with open(os.path.join(restore_path, "checkpoint.txt")) as f:
                    lines = f.readlines()
                for line in lines:
                    if "epoch" in line:
                        counter = int(line.split(" ")[1].strip())
            elif os.path.exists(os.path.join(restore_path, "checkpoint.yaml")):
                data = OmegaConf.load(
                    os.path.join(restore_path, "checkpoint.yaml"))
                counter = data["epoch"]

        checkpoint_path = os.path.join(
            restore_path, "last_checkpoint_{}.tar".format("%04d" % counter))

    checkpoint = load_checkpoint(checkpoint_path, rank)
    epoch_check = checkpoint["epoch"]

    model.load_state_dict(checkpoint["state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optim_dict"])

    if lr_scheduler is not None and "scheduler_dict" in checkpoint.keys():
        lr_scheduler.load_state_dict(checkpoint["scheduler_dict"])

    # sanity check that copy is done correctly.
    for name, param in model.named_parameters():
        keys = checkpoint["state_dict"].keys()
        if name in keys:
            assert torch.allclose(param.data, checkpoint["state_dict"][name])

    return model, optimizer, lr_scheduler, epoch_check, is_partial


class RunningAverage:
    """A simple class that maintains the running average of a quantity
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps) if self.steps != 0 else 0


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, "w") as f:
        # We need to convert the values to float for json
        # (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, checkpoint, counter=0):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'.
    If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such
        as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint,
                            "last_checkpoint_{}.tar".format(counter))
    if not os.path.exists(checkpoint):
        logger.info(
            "Checkpoint Directory does not exist! Making directory {}".format(
                checkpoint))
        os.mkdir(checkpoint)
    else:
        logger.info("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    return filepath


def load_checkpoint(checkpoint_path, rank=0):
    """Loads model parameters (state_dict) from file_path.
    If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint_path: (string) filename which needs to be loaded
    """

    if not os.path.exists(checkpoint_path):
        raise ValueError("File doesn't exist: {}".format(checkpoint_path))

    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}

    logger.info("Restoring parameters from {}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    return checkpoint

