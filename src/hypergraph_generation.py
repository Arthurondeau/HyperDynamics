import os
import hydra
from omegaconf import DictConfig
from src.utils.hypergraph_utils import create_hypergraph
import xgi 


@hydra.main(config_path="../conf", config_name="defaults", version_base="1.1")
def generate_hypergraph(cfg: DictConfig) -> xgi.Hypergraph:
    """
    Generate a hypergraph based on a configuration file and save it.

    Args:
        cfg (DictConfig): Configuration object loaded by Hydra.
    """
    # Extract parameters from the configuration
    hypergraph_name = cfg.networks.network_type
    output_dir = cfg.networks.output_dir

    # Generate the hypergraph
    hypergraph = create_hypergraph(cfg.networks.input_dir)
    print(f"Hypergraph generated with type: {cfg.networks.network_type}")
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define the output file path
    output_file = os.path.join(output_dir, hypergraph_name + '.json')

    # Save the hypergraph to a file
    with open(output_file, 'w') as f:
        xgi.write_json(hypergraph, hypergraph_name + '.json')


    print(f"Hypergraph saved to {output_file}")

    return hypergraph