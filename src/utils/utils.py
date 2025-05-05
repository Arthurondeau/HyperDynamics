import numpy as np


def EDVW_weights(H:object,EDVW_type:str)-> object:
    """
    Computes the  matrix weight R for the EDVW (Edge-Driven View Weight) based on the given network configuration.

    Args:
        H (Hypergraph object): Hypergraph class object.
        EDVW_type (str): Type of distribution to use for generating weights. Currently only 'normal' is supported.

    Returns:
        R: matrix array of weights for the hyperedges. shape is (M,N) with M number of hyperedges and N number of nodes.
    """
    R = np.ones((len(H.edges), len(H.nodes)))

    if EDVW_type == 'normal':
        for i, e in enumerate(H.edges):
            for node in H.edges.members(e):
                R[i, node] = np.random.normal(0.5, 0.1)
    return R