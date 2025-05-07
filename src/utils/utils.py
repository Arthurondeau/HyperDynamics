import numpy as np
import xgi


def EDVW_vertex_weights(H: xgi.Hypergraph, R_type: str) -> np.ndarray:
    """
    Compute the vertex weight matrix R for the EDVW hypergraph based on work by
    U Chitra, B Raphael, "Random walks on hypergraphs with edge-dependent vertex weights" 2019. 
    This function calculates a matrix `R` where each entry represents the weight of a vertex
    in a specific hyperedge. The weights are generated based on the specified distribution type.

    Parameters:
    -----------
    H : xgi.Hypergraph
        The input hypergraph, represented as an XGI hypergraph object.
    R_type : str
        The type of distribution to use for generating weights. Currently, only 'normal'
        distribution is supported.

    Returns:
    --------
    numpy.ndarray
        A 2D matrix `R` of shape (M, N), where `M` is the number of hyperedges and `N` is the
        number of nodes. Each entry `R[i, j]` represents the weight of node `j` in hyperedge `i`.

    Notes:
    ------
    - The weights are generated using a normal distribution with mean 0.5 and standard
      deviation 0.1.
    - If a node does not belong to a hyperedge, its weight in that hyperedge is set to 0.
    """
    R = np.zeros((len(H.edges), len(H.nodes)))

    if R_type == 'normal':
        for i, edge in enumerate(H.edges):
            for node in H.edges.members(edge):
                R[i, node] = np.random.normal(0.5, 0.1)

    return R


def Hyperedge_weights(H:object,W_weights_type:str,order:int)-> np.ndarray:
    """
    Compute the hyperedge weight matrix W for the hypergraph.
    This function calculates a matrix `W` where each entry represents the weight of a node
    in a specific hyperedge. The weights are generated based on the specified distribution type.

    Parameters:
    -----------
    H : xgi.Hypergraph
        The input hypergraph, represented as an XGI hypergraph object.
    W_weights_type : str
        The type of distribution to use for generating weights. Currently, only 'normal'
        distribution is supported.
    order : int
        The order of the hypergraph (not currently used in the function).

    Returns:
    --------
    numpy.ndarray
        A 2D matrix `W` of shape (N, M), where `N` is the number of nodes and `M` is the
        number of hyperedges. Each entry `W[i, j]` represents the weight of node `i` in
        hyperedge `j`.

    Notes:
    ------
    - The weights are generated using a normal distribution with mean 0.5 and standard
      deviation 0.1.
    - If a node does not belong to a hyperedge, its weight in that hyperedge is set to 0.
    """
    ## Initialize the weight matrix (default is zero is a node doesn't belong to a hyperedge)
    W = np.zeros((len(H.nodes), len(H.edges)))

    for edge in H.edges:
        if W_weights_type == 'normal':
            weight = np.random.normal(0.5, 0.1)
            H.edges[edge]['weight'] = weight
            for node in H.edges.members(edge):
                    W[node,edge] = weight #Common weight for all nodes in the hyperedge

    return W

def lambda_normal_weights(node:int,edge:int,H:xgi.hypergraph)-> float:
    """
    Computes the lambda normal weights for the hypergraph based on the given network configuration.

    Args:
        node (int): Node index.
        edge (int): Edge index.
        H (XGI hypergraph object): Hypergraph input.
    Returns:
        weight: Weight for the (node,hyperedge).
    """

    weight = np.random.normal(0.5, 0.1, size=(node, edge))
    return weight

def compute_gh_adj(R, W)-> np.ndarray:
    """
    Compute the generalized adjcency matrix of the clique graph based on Eq 10 from 
    U Chitra, B Raphael, "Random walks on hypergraphs with edge-dependent vertex weights" 2019. 
    This function calculates a vertex-to-vertex weight matrix `A` for a hypergraph
    based on the incidence matrix `R` and the edge weight matrix `W`. The resulting
    matrix `A` represents the weighted connections between vertices in the hypergraph.
    Parameters:
    -----------
    R : numpy.ndarray
        A 2D binary incidence matrix of shape (E, V), where `E` is the number of edges
        and `V` is the number of vertices. `R[e, v]` is non-zero if vertex `v` belongs
        to edge `e`.
    W : numpy.ndarray
        A 2D weight matrix of shape (V, E), where `W[v, e]` represents the weight of
        vertex `v` in edge `e`.
    Returns:
    --------
    numpy.ndarray
        A 2D symmetric matrix `A` of shape (V, V), where `A[u, v]` represents the
        weighted connection between vertices `u` and `v` in the hypergraph.
    Notes:
    ------
    - The edge weight vector `WE` is computed by extracting the first non-zero weight
      greater than zero for each edge in `W`.
    - The matrix `A` is constructed by iterating over all edges and summing the
      contributions of each edge to the vertex-to-vertex weights.
    Example:
    --------
    Given an incidence matrix `R` and weight matrix `W`, compute the vertex-to-vertex
    weight matrix `A`:
    >>> R = np.array([[1, 0, 1], [0, 1, 1]])
    >>> W = np.array([[1, 0], [0, 2], [3, 0]])
    >>> A = compute_gh_weights(R, W)
    >>> print(A)
    [[10.  3. 10.]
     [ 3.  4.  6.]
     [10.  6. 13.]]
    """
    E, V = R.shape
    A = np.zeros([V,V]) # to return
    
    # first, create edge weight vector
    WE = np.zeros(E)
    # for each edge, find first non-zero value that is >0
    for e in range(E):
        WE[e] = W[np.where(W[:,e] > 0)[0],e][0]
    
    # iterate over edges, add w(e) * gam_e(u) * gam_e(v) term
    # for each pair of vertices u,v \in e
    for e in range(E):
        nodes_in_e = np.nonzero(R[e,:])[0]
        for u in nodes_in_e:
            for v in nodes_in_e:
                A[u,v] += WE[e] * R[e,u] * R[e,v]
    return A


