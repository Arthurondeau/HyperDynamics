import os
import hydra
from omegaconf import DictConfig

from src.networks import instantiate_network

# Chitra_Paper.py
"""
Script for analysis and experiments related to the Chitra Paper.
This script is part of the HyperDynamics project.
"""


@hydra.main(config_path="../conf", config_name="defaults", version_base="1.1")
def main(cfg: DictConfig):
    """
    Main function to execute the script.
    
    Args:
        cfg (DictConfig): Configuration object loaded by Hydra.
    """
    ## Instantiate the network
    network = instantiate_network(cfg)
    print(f"Network instantiated with type: {cfg.networks.network_type}")
    ## Project the network
    #clique_network = network.project_to_clique()
# Entry point of the script
if __name__ == "__main__":
    main()