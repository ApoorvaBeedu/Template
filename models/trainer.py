import copy
import datetime
import os
from abc import abstractmethod

import numpy as np
import torch
import torch.multiprocessing
import wandb
from loguru import logger
from omegaconf import OmegaConf, open_dict
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import pytorch_utils as util
from utils.ddp_utils import convert_groupnorm_model

import models.optimiser as optimiser
from models.enums import Losses
from models.loss import Loss


def get_model(args):
    """Use this to chose from different models you may have created

    Args:
        args (_type_): _description_

    Raises:
        NotImplementedError: _description_
    """
    raise NotImplementedError()


class GenericTrainer:

    def __init__(self, args):
        super(GenericTrainer).__init__()

        self.model = get_model(args)
        # Set optimiser and the scheduler
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optimiser.get_optimizer(args, params)
        
        self.lr_scheduler = None
        self.criterion = Loss(args)
        self.args = args
        self.losses = Losses

    def model(self) -> torch.nn.Module:
        return self.model

    def writer(self) -> util.WriteLogs():
        if not self.writer:
            return None
        return self.writer
    
    def set_scheduler(self):
        self.lr_scheduler = optimiser.get_scheduler(self.args, self.optimizer)

    def write_args_to_file(self):
        with open(self.args.model_dir + "/arguments.yaml", "w") as f:
            OmegaConf.save(config=self.args, f=f)

    @abstractmethod
    def initialisation(self):
        """Use this to initialise all the required folders, wandb and other stuff

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError

    @abstractmethod
    def try_init_trainer(self, rank,):
        """Use this to load pretrained weights, and initialise the model

        Raises:
            NotImplementedError: _description_
        """
        args = self.args
        self.model = convert_groupnorm_model(self.model)
        self.model = self.model.to(args.device)
        self.model = DDP(self.model,
                         device_ids=[args.device],
                         output_device=args.device,
                         find_unused_parameters=True)
        if rank == 0:
            pass # retore_from_file
        raise NotImplementedError

    def step(self):
        """This is called after every iteration
        """
        self.optimizer.step()
        self.lr_scheduler.step()

    def zero_grad(self):
        self.model.zero_grad()
        self.optimizer.zero_grad()

    @abstractmethod
    def on_epoch_start(self):
        """Implement all the things that need to happen at the start of the epoch

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError

    @abstractmethod
    def on_epoch_complete(self):
        """Implement all that needs to happen at the end of the epoch

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError

    @abstractmethod
    def on_iteration_complete(self):
        """Implement everything that needs to happen at the end of an iteration if need be.
        """
        raise NotImplementedError

    @abstractmethod
    def forward_impl(self):
        """Call the forward pass for the model here. F

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError

