import torch.optim as optim
from torch.optim.lr_scheduler import (CosineAnnealingWarmRestarts, CyclicLR,
                                      LambdaLR, LinearLR, OneCycleLR,
                                      ReduceLROnPlateau, StepLR)


def get_optimizer(args, optim_params):
    """
    Initialises and returns an optimizer
    :param args:
    :param optim_params: model optim parameters
    :return: optimizer
    """
    if args.optimiser.optimiser == "adam":
        optimizer = optim.Adam(optim_params,
                               lr=args.optimiser.lr,
                               eps=1e-4,
                               weight_decay=args.optimiser.weight_decay)
    else:
        optimizer = optim.SGD(
            optim_params,
            lr=args.optimiser.lr,
            nesterov=True,
            weight_decay=args.optimiser.weight_decay,
            momentum=0.9,
            dampening=0,
        )
    return optimizer


def get_scheduler(args, optimizer):
    if "step" in args.optimiser.scheduler_type:
        lr_scheduler = StepLR(
            optimizer,
            step_size=args.optimiser.scheduler_stepSize,
            gamma=args.optimiser.lr_multiplier,
            verbose=True,
        )
    elif "reduce" in args.optimiser.scheduler_type:
        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            factor=args.optimiser.lr_multiplier,
            patience=3,
            threshold=0.1,
            verbose=True,
        )
    elif "lambda" in args.optimiser.scheduler_type:
        lr_decay_lambda = lambda epoch: args.optimiser.lr_multiplier**(
            epoch // args.optimiser.scheduler_stepSize)
        lr_scheduler = LambdaLR(
            optimizer,
            lr_lambda=lr_decay_lambda,
            verbose=True,
        )
    elif "onecyclelr" in args.optimiser.scheduler_type:
        lr_scheduler = OneCycleLR(
            optimizer,
            max_lr=args.optimiser.max_lr,
            epochs=args.optimiser.num_epochs,
            anneal_strategy=args.optimiser.anneal_strategy,
            verbose=True,
        )
    elif "cycliclr" in args.optimiser.scheduler_type:
        lr_scheduler = CyclicLR(
            optimizer,
            base_lr=args.optimiser.lr,
            max_lr=args.optimiser.max_lr,
            step_size_up=args.optimiser.step_size_up,
            mode=args.optimiser.anneal_strategy,
            verbose=True,
        )
    elif "cosineannealingwarm" in args.optimiser.scheduler_type:
        lr_scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.optimiser.T_0,
            T_mult=args.optimiser.T_mult,
            verbose=True,
        )
    elif "none" in args.optimiser.scheduler_type:
        lr_scheduler = None
    else:
        raise NotImplementedError
    return lr_scheduler


def adjust_learning_rate(optimizer, multiplier, stop_lr):
    """Sets the learning rate to the initial LR decayed by multiplier when
    called
    :param optimizer: optimizer used
    :param multiplier: LR is decreased by this factor
    """
    for param_group in optimizer.param_groups:
        newlr = param_group["lr"] * multiplier
        if newlr > (stop_lr):
            param_group["lr"] = newlr
