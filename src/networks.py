import importlib
from omegaconf import DictConfig
import hydra
from src.utils.utils import EDVW_weights


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

    ## Define hyperedge vertex weight distribution
    if H.EDVW:
        H.R = EDVW_weights(H.graph, EDVW_type=cfg.networks.EDVW_params.EDVW_type)
    print(f"EDVW Weights instantiated with type: {cfg.networks.EDVW_params.EDVW_type}")
    return H


