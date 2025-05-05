import xgi 

class RandomHyperGraph(object):
    """
    Generates a random hypergraph
    Generate N nodes, and connect any d+1 nodes by a hyperedge with probability ps[d-1].
    """

    def __init__(self, n: int, ps: list[float], order: int, seed: int,EDVW: bool):
        self.graph = xgi.random_hypergraph(n, ps, order=order, seed=seed)
        self.EDVW = EDVW
        