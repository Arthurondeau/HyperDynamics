import importlib
from omegaconf import DictConfig
import hydra
from src.utils.utils import EDVW_vertex_weights,Hyperedge_weights, compute_gh_adj
import numpy as np
import xgi 


def instantiate_network(cfg: DictConfig):
    """
    Instantiates a network based on the configuration provided using Hydra's instantiate method.

    Args:
        cfg (DictConfig): Configuration object containing network details.

    Returns:
        object: Instantiated network object.
    """

    network_type = cfg.networks.network_type
    ## Instantiate the network from the config file with XGI.Generator
    H = hydra.utils.instantiate(cfg.networks[network_type])
    print(f"Network instantiated with type: {cfg.networks.network_type}")

    ## Define vertex-weight matrix R
    if H.EDVW:
        H.R = EDVW_vertex_weights(H.graph, R_type=cfg.networks.EDVW_params.R_type)
    print(f"EDVW Weights instantiated with type: {cfg.networks.EDVW_params.R_type}")

    ## Define hyperedge-weight matrix W
    if H.weighted:
        H.W = Hyperedge_weights(H.graph, W_weights_type=cfg.networks.W_weights_type,order=cfg["networks"][network_type]["order"])

    ## Define degree matrixes
    # vertex-degree matrix Dv
    H.Dv = np.diag(np.sum(H.W, axis=1))
    # hyperedge-degree matrix De
    H.De = np.diag(np.sum(H.R, axis=1))

    ## Compute the projected clique graph
    if cfg.networks.project_clique:
        clique_network = compute_gh_adj(H.R, H.W)
        print(f"Projected clique graph with type: {cfg.networks.project_clique}")
        print(f"Projected clique graph shape: {clique_network}")
    return H


