from logging import logger

import cv2
import numpy as np
import torch

import utils.pytorch_utils as util


### Writing to log stuff
def writer_log_text(phase, optimizer, epoch, writer):
    lr_string = "{0} <br> lr: ".format(phase)
    for param_group in optimizer.param_groups:
        logger.info("Current lr: {0}".format(param_group["lr"]))
        lr_string += "{0} ".format(param_group["lr"])
    writer.log_text(
        "{0}_losses".format(phase),
        "{0} <br> epoch: {1}".format(lr_string, epoch),
    )


class WriteLogs:

    def __init__(self, writer=None, wandb=None):
        """
            Writer -> Tensorboard
            Wandb -> Weights and biases writer
        """
        self.writer = writer
        self.wandb = wandb

    def log_scalars_dict(self, key, val, step=0):
        if self.writer is not None:
            self.writer.add_scalars(key, val, step)
        if self.wandb is not None:
            self.wandb.log({key: val})

    def log_scalar(self, key, val, step=0):
        if self.writer is not None:
            self.writer.add_scalar(key, val, step)
        if self.wandb is not None:
            self.wandb.log({key: val})

    def log_images(self, key, image, step=0):
        if image is None:
            return
        if self.writer is not None:
            self.writer.add_image(key, image, global_step=0)
        if self.wandb is not None:
            if len(image.shape) > 2:
                image = image.transpose(1, 2, 0)
            self.wandb.log({key: [self.wandb.Image(image, caption=key)]})

    def log_text(self, key, text, step=0):
        if self.writer is not None:
            self.writer.add_text(key, text)
        if self.wandb is not None:
            self.wandb.log({key: text})

