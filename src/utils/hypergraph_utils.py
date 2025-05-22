import xgi
import pandas as pd
import os

#     ----------------------------------------------------------

def create_hypergraph(path_folder: str) -> xgi.Hypergraph : 

    """
    Generate a hypergraph from a file.

    Args:
        path_file (str): Path to the folder containing the csv data.

    Returns:
        xgi.Hypergraph: The generated hypergraph.
    """
    # Find the CSV file in the folder
    csv_files = [f for f in os.listdir(path_folder) if f.endswith('.csv')]
    if len(csv_files) != 1:
        raise ValueError("The folder must contain exactly one CSV file.")
    
    # Load the hypergraph from the file
    file_path = os.path.join(path_folder, csv_files[0])
    dataframe = pd.read_csv(file_path)
    
    countries_groups = []
    # Create a hypergraph where each EventID corresponds to a hyperedge
    for event_id, group in dataframe.groupby("EventID"):
        countries = group[["Country1", "Country2"]].values.flatten()
        countries = [country for country in countries if pd.notna(country)]  # Remove NaN values
        countries_groups.append(countries)

    # Create the hypergraph 
    H = xgi.Hypergraph()
    # add hyperedges (countries) to the hypergraph
    H.add_edges_from(countries_groups)
    print(f"Hypergraph created with {len(countries_groups)} hyperedges.")
    return H 
