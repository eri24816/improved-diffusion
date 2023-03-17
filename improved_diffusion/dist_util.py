"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
try:
    from mpi4py import MPI
    USE_DIST = True
    print("Using MPI.")
except ModuleNotFoundError:
    USE_DIST = False
    print("Not using MPI.")

import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3


def setup_dist():
    """
    Setup a distributed process group.
    """
    if not USE_DIST:
        return
    if dist.is_initialized():
        return

    comm = MPI.COMM_WORLD
    backend = "gloo" if not th.cuda.is_available() else "nccl"

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())
    os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    os.environ["RANK"] = str(comm.rank)
    os.environ["WORLD_SIZE"] = str(comm.size)

    port = comm.bcast(_find_free_port(), root=0)
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=backend, init_method="env://")

def dev():
    """
    Get the device to use for torch.distributed.
    """
    if not USE_DIST:
        return th.device("cuda" if th.cuda.is_available() else "cpu")
    if th.cuda.is_available():
        return th.device(f"cuda:{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    if not USE_DIST:
        return th.load(path, **kwargs)
    if MPI.COMM_WORLD.Get_rank() == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
    else:
        data = None
    data = MPI.COMM_WORLD.bcast(data)
    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    if not USE_DIST:
        return
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()

# wrap some torch.distributed functions to make them no-ops if we're not using dist

def all_reduce(*args, **kwargs):
    if USE_DIST:
        return dist.all_reduce(*args, **kwargs)
    return None

def all_gather(*args, **kwargs):
    if USE_DIST:
        return dist.all_gather(*args, **kwargs)
    return None 

def reduce(*args, **kwargs):
    if USE_DIST:
        return dist.reduce(*args, **kwargs)
    return None

def get_rank():
    if USE_DIST:
        return dist.get_rank()
    return 0

def get_world_size():
    if USE_DIST:
        return dist.get_world_size()
    return 1

def barrier():
    if USE_DIST:
        return dist.barrier()
    return None

def broadcast(*args, **kwargs):
    if USE_DIST:
        return dist.broadcast(*args, **kwargs)
    return None

def is_initialized():
    if USE_DIST:
        return dist.is_initialized()
    return False
