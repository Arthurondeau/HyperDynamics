import os
import hydra
from omegaconf import DictConfig

from src.networks import instantiate_network
from src.dataset_generation import generate_dataset
from src.hypergraph_generation import generate_hypergraph

# Chitra_Paper.py
"""
Script for the pipeline to generate hypergraph for dataset, compute the community detection on the
following hypergraph.
"""


@hydra.main(config_path="../conf", config_name="defaults", version_base="1.1")
def main(cfg: DictConfig):
    """
    Main function to execute the script.
    
    Args:
        cfg (DictConfig): Configuration object loaded by Hydra.
    """
    ## Generate the dataset 
    generate_dataset(cfg)

    ## Generate the hypergraph
    H = generate_hypergraph(cfg)
    print(f"Hypergraph instantiated from dataset: {cfg.networks.dataset_name}")

    ## Compute Community Detection

    
    #clique_network = network.project_to_clique()
# Entry point of the script
if __name__ == "__main__":
    main()