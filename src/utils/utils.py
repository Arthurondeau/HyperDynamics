import numpy as np
import xgi
import random
from pprint import pprint

import numpy as np
import scipy.optimize
import scipy.stats


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

def compute_gh_adj(R:np.ndarray, W:np.ndarray)-> np.ndarray:
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


def random_walk_P(R:np.ndarray,W:np.ndarray) -> int:
    """
    Compute the random walk matrix P for the hypergraph based on the incidence matrix R and
    the edge weight matrix W. The resulting matrix P represents the transition probabilities
    for a random walk on the hypergraph.

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
        A 2D transition probability matrix P of shape (V, V), where P[u, v] represents
        the probability of transitioning from vertex u to vertex v in one step of the
        random walk.

    Notes:
    ------
    - The transition probabilities are computed based on the edge weights and the incidence
      matrix.
    """
    
    
    # create diagonal matrix with entries d(v)
    D_V = degree_matrix(W)

    # create prob trans matrix
    P = np.linalg.inv(D_V).dot(W).dot(R)
    P = np.transpose(P) # since we're using column vectors

    return P

def degree_matrix(X:np.ndarray)-> np.ndarray:
    """
    Compute the degree matrix for a given incidence matrix X.
    The degree matrix is a diagonal matrix where each diagonal element
    represents the degree of the corresponding vertex/edge in the hypergraph.
    Parameters:
    -----------
    X : numpy.ndarray
        A 2D binary incidence matrix of shape (E, V) (or (V,E)), where `E` is the number of edges
        and `V` is the number of vertices. `X[e, v]` is non-zero if vertex `v` belongs
        to edge `e`.
    Returns:
    --------
    numpy.ndarray
        A 2D diagonal matrix of shape (V, V) (or (E,E)), where the diagonal elements represent
        the degree of each vertex/edge in the hypergraph.
    """

    # Compute the degree of each vertex/edge
    return np.diag(np.sum(X, axis=1))



##################################################
# HELPERS FUNCTION RELATED TO CHITRA PAPER
##################################################


def create_pi_list(n, s, p, means, sigma):
    pi_list = []
    included_elts = set() # list of all elements in some partial ranking
    while (len(included_elts) < n) or (len(pi_list) < s):
        pi=[]
        for v in range(1, n+1):
            if random.uniform(0,1) < p:
                pi += [v]
        scaling = np.random.uniform(low=1/3,high=3)
        scores = [np.random.normal(means[w-1] / 5, sigma) * scaling for w in pi]

        if len(pi) > 1:
            scores_sorted = [x for x, _ in sorted(zip(scores, pi), key=lambda pair: pair[0])]
            pi_sorted = [x for _, x in sorted(zip(scores, pi), key=lambda pair: pair[0])]
            pi_list += [(pi_sorted, scores_sorted)]
            included_elts.update(pi_sorted)
    universe=np.array(list(included_elts))
    return (universe, pi_list)


##################################################
# given probability transition matrix P
# where P_{v,w} = Prob(w -> v)
# find pagerank scores with restart probability r
def compute_pr(P:np.ndarray, r:float, n:int, eps:float=1e-8)-> np.ndarray:
    """
    Compute PageRank scores for a given probability transition matrix P.
    This function calculates the PageRank scores using the power iteration method with
    a restart probability of r. The scores are computed until convergence or until
    a maximum number of iterations is reached.
    Parameters:
    -----------
    P : numpy.ndarray
        A 2D probability transition matrix of shape (V, V), where P[u, v] represents
        the probability of transitioning from vertex u to vertex v in one step of the
        random walk.
    r : float
        The restart probability for the random walk. A value between 0 and 1.
    n : int
        The number of nodes in the hypergraph.
    eps : float, optional
        The convergence threshold for the PageRank scores. Default is 1e-8.
    Returns:
    --------
    numpy.ndarray
        A 1D array of PageRank scores for each node in the hypergraph. The scores are
        normalized to sum to 1.
    Notes:
    ------
    - The PageRank scores are computed using the power iteration method with a restart
      probability of r.
    - The scores are normalized to sum to 1.
    Example:
    --------
    Given a probability transition matrix P and a restart probability r, compute
    the PageRank scores:
    >>> P = np.array([[0.1, 0.9], [0.9, 0.1]])
    >>> r = 0.15
    >>> n = 2
    >>> scores = compute_pr(P, r, n)
    >>> print(scores)
    [0.5 0.5]
    """
    x = np.ones(n) / n*1.0
    flag = True
    t=0
    while flag:
        x_new = (1-r)*P.dot(x)
        x_new = x_new + np.ones(n) * r / n
        diff = np.linalg.norm(x_new - x)
        if np.linalg.norm(x_new - x,ord=1) < eps and t > 100:
            flag = False
        t=t+1
        x = x_new
    return x

def hg_rank(universe, pi_list):
    # first create these matrices
    # R = |E| x |V|, H(e, v) = lambda_e(v)
    # W = |V| x |E|, W(v, e) = w(e) 1(v in e)
    m = len(pi_list) # number of hyperedges
    n = len(universe) # number of items to be ranked 
    R = np.zeros([m, n])
    W = np.zeros([n, m])

    for i in range(len(pi_list)):
        pi, scores = pi_list[i]
        if len(pi) > 1:   
            for j in range(len(pi)):
                v = pi[j]
                v = np.where(universe == v)[0][0] #equivalent to universe.index(v) but for np arrays
                R[i, v] = np.exp(scores[j])
                W[v, i] = 1.0

            # edge weight is stdev of vertex weights
            W[:, i] = (np.std(scores) + 1.0) * W[:, i]

            R[i, :] = R[i,:] / sum(R[i,:])

    # create diagonal matrix with entries d(v)
    D_V = degree_matrix(W)

    # create prob trans matrix
    P = random_walk_P(R, W)
    # create RWR matrix
    r=0.40
    rankings = compute_pr(P, r, n)
    return universe[np.argsort(rankings)]

def gh_rank(universe, pi_list):
    # first create these matrices
    # R = |E| x |V|, H(e, v) = lambda_e(v)
    # W = |V| x |E|, W(v, e) = w(e) 1(v in e)
    m = len(pi_list) # number of hyperedges
    n = len(universe) # number of items to be ranked 
    R = np.zeros([m, n])
    W = np.zeros([n, m])

    for i in range(len(pi_list)):
        pi, scores = pi_list[i]
        if len(pi) > 1:   
            for j in range(len(pi)):
                v = pi[j]
                v = np.where(universe == v)[0][0] #equivalent to universe.index(v) but for np arrays
                R[i, v] = np.exp(scores[j])
                W[v, i] = 1.0

            # edge weight is stdev of vertex weights?
            W[:, i] = (np.std(scores) + 1.0) * W[:, i]

            R[i, :] = R[i,:] / sum(R[i,:])

    # compute edge weights of G^H
    A = compute_gh_adj(R, W)
            
    # create prob trans matrix by normalizing columns to sum to 1
    P = A/A.sum(axis=1)[:,None]
    P=P.T

    # create RWR matrix
    r=0.40
    rankings = compute_pr(P, r, n)
    return universe[np.argsort(rankings)]