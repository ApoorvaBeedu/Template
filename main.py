import os
import os.path as osp
import time
import warnings
from datetime import datetime

import hydra
import numpy as np
import torch
import torch.multiprocessing as mp
import wandb
from loguru import logger
from omegaconf import OmegaConf, open_dict

from datasets import data
from models.enums import Split
from models.trainer import Trainer
from utils import pytorch_utils as p_utils
from utils.ddp_utils import find_free_port, set_distributed

torch.multiprocessing.set_sharing_strategy("file_system")
from hydra.utils import get_original_cwd

warnings.filterwarnings("ignore")
dir_path = os.path.dirname(os.path.realpath(__file__))

CUDA_LAUNCH_BLOCKING = 1
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def get_dataloader(args) -> dict:
    """return a dict with dataloader for every split that will be used in the training

    Args:
        args (_type_): _description_

    Returns:
        dict: {"train":None, "val":None}
    """
    pass


def solve(rank, world_size, args):
    device = set_distributed(rank, world_size, args)

    loaders = get_dataloader(args)
    
    trainer = Trainer(args, loaders[list(loaders)[0]].dataset, device)
    trainer.initialisation(rank)
    start_epoch = 0
    start_epoch = trainer.try_init_trainer(rank)

    if rank == 0:
        summary_writer = trainer.writer
        wandb.watch(trainer.model)
        total_params = sum(p.numel() for p in trainer.model.parameters()
                           if p.requires_grad)
        logger.info("Total number of parameters {}".format(total_params))
        summary_writer.log_text("Arguments", "{0} <br> ".format(args))

    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        if rank == 0:
            logger.info(f"Starting Epoch {epoch}")
        for split, loader in loaders.items():
            if rank == 0:
                logger.info(f"Starting Split {split}")
            trainer.on_epoch_start(epoch, split)
            trainer.count = 0
            if split not in args.split:
                continue

            with torch.set_grad_enabled(split == Split.Train.value):
                t0 = time.time()
                for batch_idx, batch in enumerate(loader):
                    trainer.zero_grad()
                    trainer.count += 1
                    if rank == 0:
                        logger.info(
                            f"Got {batch_idx+1}/{len(loader)} batches in {split}"
                        )
                        logger.info("Forward pass")
                    t0 = time.time()
                    outputs = trainer.forward_impl(batch, rank)
                    loss = outputs["loss"]
                    torch.distributed.all_reduce(loss)

                    if "train" in split:
                        split_losses = trainer.losses.train
                    else:
                        split_losses = trainer.losses.test

                    split_losses.loss_avg.update(loss.item())
                    if rank == 0:
                        trainer.on_iteration_complete()

                    t1 = time.time()
                    if rank == 0:
                        logger.info(
                            f"Done with forward pass in {t1-t0} seconds")
                    if split != Split.Train.value:
                        continue
                    trainer.zero_grad()
                    if rank == 0:
                        logger.info("Backward pass")
                    loss.backward()
                    if rank == 0:
                        logger.info("Done backward pass")
                    trainer.step(epoch + batch_idx / len(loader))
                    t2 = time.time()
                    if rank == 0:
                        logger.info(
                            "{0}ing batch index/total {2}/{3} epoch {4} in time: {5} \nLoss Avg: {1} \n"
                            .format(split, loss.item(), batch_idx, len(loader),
                                    epoch, (t2 - t0)))
                    trainer.num_train_batches += 1
                if rank == 0:
                    logger.info(f"{split} has finished")
                    if "train" in split:
                        trainer.save_checkpoint()
        if rank == 0:
            trainer.on_epoch_complete()


def initialise(args):
    # Checkpoint stuff, creating of directories that are project specific
    if not osp.isdir(args.model_dir_path):
        os.mkdir(args.model_dir_path)

    # Restore from file, and if arguments.yaml exist, load the arguments to avoid any training/test discrepancies.
    if args.restore_file is not None and args.use_config:
        config_path = os.path.join(args.model_dir_path, args.restore_file,
                                   "arguments.yaml")
        with open(config_path, "r") as f:
            logger.info(f"loading config file from {config_path}")
            conf_temp = OmegaConf.load(f)
            del conf_temp["restore_file"]
        with open_dict(args):
            for k, v in args.items():
                if k in conf_temp.keys():
                    args[k] = conf_temp[k]

    # Creates a path for checkpoint saving based on wandb env_name.
    path = "{}_lr_{}_wd_{}".format(args.wandb.env_name, str(args.optimiser.lr),
                                   str(args.optimiser.weight_decay))

    checkpoint_dir_path = osp.join(args.model_dir_path, path)
    if not osp.isdir(checkpoint_dir_path):
        os.mkdir(checkpoint_dir_path)
    model_dir = checkpoint_dir_path

    with open_dict(args):
        args.model_dir = model_dir
        args.writer_dir = os.path.join("./logs", args.wandb.env_name)


@hydra.main(config_path="./config", config_name="config")
def main(args):
    root = args.cwd
    with open_dict(args):
        args.root_path = dir_path  #upto EPIC
        args.data_path = os.path.join(dir_path, args.data_path)
        args.cwd = root
    initialise(args)
    logger.start(args.model_dir + "/bash_logger.log")
    logger.info(args)

    # Setup seeds
    np.random.seed(42)
    torch.manual_seed(500)
    torch.backends.cudnn.deterministic = True
    device = "cpu"

    start_time = datetime.now()
    logger.info("Time: {0}".format(start_time))

    # Train the model
    logger.info("Starting training for {} epoch(s)".format(args.num_epochs))
    epoch_start = 1

    with open_dict(args):
        args.master_port = find_free_port()
        cuda_device_cnt = torch.cuda.device_count()
        args.world_size = args.num_gpu if args.num_gpu > 0 else cuda_device_cnt

    mp.spawn(solve,
             args=(args.world_size, args),
             nprocs=args.world_size,
             join=True)
    # solve(loaders, trainer, args.num_epochs, args)
    logger.info("Finished")


if __name__ == "__main__":
    main()
