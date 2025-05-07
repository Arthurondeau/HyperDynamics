import importlib
from omegaconf import DictConfig
import hydra
from src.utils.utils import EDVW_vertex_weights,Hyperedge_weights, compute_gh_adj
import numpy as np
import xgi 


def random_walk(cfg: DictConfig, H, steps, start_node=None):
    """
    Wrapper to call the random walk function on the network, using configuration details.

    Args:
        cfg (DictConfig): Configuration object containing random walk parameters.
        H (object): The network object.
        steps (int): Number of steps for the random walk.
        start_node (int, optional): Starting node for the random walk. If None, a random node is chosen.

    Returns:
        list: Sequence of nodes visited during the random walk.
    """
    random_walk_type = cfg.random_walk.type
    print(f"Performing random walk of type: {random_walk_type}")
    return perform_random_walk(H.graph, steps, start_node, walk_type=random_walk_type)
