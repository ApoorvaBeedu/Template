import os
import signal
import threading

import ifcfg
import torch
import torch.distributed as distrib
import torch.nn as nn
from loguru import logger
from omegaconf import OmegaConf, open_dict

# EXIT = threading.Event()
# EXIT.clear()

# def _clean_exit_handler(signum, frame):
#     EXIT.set()
#     print("Exiting cleanly", flush=True)

# signal.signal(signal.SIGINT, _clean_exit_handler)
# signal.signal(signal.SIGTERM, _clean_exit_handler)
# signal.signal(signal.SIGUSR2, _clean_exit_handler)


def get_ifname():
    return ifcfg.default_interface()["device"]


def init_distrib_slurm(backend="nccl"):
    if "GLOO_SOCKET_IFNAME" not in os.environ:
        os.environ["GLOO_SOCKET_IFNAME"] = get_ifname()

    if "NCCL_SOCKET_IFNAME" not in os.environ:
        os.environ["NCCL_SOCKET_IFNAME"] = get_ifname()

    master_port = int(os.environ.get("MASTER_PORT", 8738))
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    local_rank = int(
        os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
    world_rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    world_size = int(
        os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))

    tcp_store = distrib.TCPStore(master_addr, master_port, world_size,
                                 world_rank == 0)
    distrib.init_process_group(backend,
                               store=tcp_store,
                               rank=world_rank,
                               world_size=world_size)

    return local_rank, tcp_store


def convert_groupnorm_model(module, ngroups=32):
    mod = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        mod = nn.GroupNorm(ngroups, module.num_features, affine=module.affine)
    for name, child in module.named_children():
        mod.add_module(name, convert_groupnorm_model(child, ngroups))

    return mod


def find_free_port():
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


def is_port_in_use(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def set_distributed(rank, world_size, args):
    with open_dict(args):
        args.master_port = int(os.environ.get("MASTER_PORT", args.master_port))
        args.master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    if rank == 0:
        logger.info(f"{args.master_addr=} {args.master_port=}")
    tcp_store = distrib.TCPStore(args.master_addr, args.master_port,
                                 world_size, rank == 0)
    distrib.init_process_group('nccl',
                               store=tcp_store,
                               rank=rank,
                               world_size=world_size)

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
    else:
        assert world_size == 1
        device = torch.device("cpu")
    return device


def is_dist_avail_and_initialized():
    if not distrib.is_available():
        return False
    if not distrib.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return distrib.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return distrib.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)
