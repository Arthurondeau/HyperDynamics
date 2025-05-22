import os
import hydra
from omegaconf import DictConfig
from src.utils.utils import load_data

@hydra.main(config_path="../conf", config_name="defaults", version_base="1.1")
def generate_dataset(cfg: DictConfig):
    """
    Generate a synthetic dataset based on a configuration file and save it as a CSV file.

    Args:
        cfg (DictConfig): Configuration object loaded by Hydra.
    """
    # Extract parameters from the configuration

    dataset_name = cfg.datasets.dataset_name
    output_dir = cfg.datasets.output_dir
     
    dataset = load_data(dataset_name, output_dir=output_dir)
    
    os.makedirs(output_dir, exist_ok=True)

    # Define the output file path
    output_file = os.path.join(output_dir, dataset_name +'.csv')

    # Save the DataFrame to a CSV file
    dataset.to_csv(output_file, index=False)

    print(f"Dataset saved to {output_file}")

