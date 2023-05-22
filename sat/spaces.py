from gym import Space
import numpy as np


class Graph(Space):
    def __init__(self, low=0, high=150, level=4, shape=None, dtype=np.float32):
        super(Graph, self).__init__(shape, dtype)

    def sample(self):
        """Randomly sample an element of this space. Can be 
        uniform or non-uniform sampling based on boundedness of space."""
        raise NotImplementedError
