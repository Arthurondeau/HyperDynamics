import importlib
from omegaconf import DictConfig
import hydra
from src.utils.utils import EDVW_vertex_weights,Hyperedge_weights, compute_gh_adj, random_walk_P
import numpy as np
import xgi 


def communication(cfg: DictConfig, H, steps, start_node=None):
    """
    Executes a random walk on the network and facilitates communication among nodes.

    Args:
        cfg (DictConfig): Configuration object containing parameters for the random walk.
        H (object): The hypergraph object containing the network structure.
        steps (int): Number of steps to perform in the random walk.
        start_node (int, optional): Starting node for the random walk. Defaults to None, in which case a random node is selected.

    Returns:
        dict: A mapping of each node to the node it communicates with after the random walk.
    """
    random_walk_type = cfg.random_walk.type
    print(f"Performing random walk of type: {random_walk_type}")
    P = random_walk_P(H.R, H.W, random_walk_type=random_walk_type)
    
    communication_mapping = {}

    for node in H.graph.nodes():
        probabilities = P[node]
        chosen_node = np.random.choice(H.graph.nodes(), p=probabilities)
        communication_mapping[node] = chosen_node
    return communication_mapping
