import numpy as np
import torch as th
import torch_geometric as pyg

from abc import ABC
from abc import abstractmethod

class Embedding(ABC):
    def __init__(self) -> None:
        self.manifold = None
        self.graph = None
        self.name = None

    # make print function print the list of attributes of the embedding, with name and type
    def __repr__(self):
        for attr in dir(self):
            if not attr.startswith('_'):
                attr_val = getattr(self, attr)
                if type(attr_val) == bool:
                    print(attr, " : ", attr_val)
                elif type(attr_val) == th.Tensor:
                    print(attr, " : th.Tensor : shape = ", attr_val.shape)
                elif type(attr_val) == str:
                    print(attr, " : ", attr_val)
                elif str(type(attr_val)) != '<class \'method\'>':
                    print(attr, " : ", type(attr_val))
        return ""

    @abstractmethod
    def __call__(self, save_to_graph=False, save_coordinates=False, return_coordinates=False, *args, **kwargs):
        """ Apply the embedding to the given graph.
        
        Args:
            save_to_graph: Whether to save the embedding to the graph object.
            save_coordinates: Whether to save the coordinates to the embedding object.
            return_coordinates: Whether to return the coordinates.
            *args: The args to the embedding. Must be specified by the implementation.
            **kwargs: The kwargs to the embedding. Idem.
        Returns:
            Depends on the implementation, but generally either a pyg.data.Data object with the embedding included, or a th.Tensor with the embedding.
        """
        raise NotImplementedError
    
    def assign_graph(self, graph: pyg.data.Data, as_ref=True) -> None:
        """ Assign a graph to the embedding.
        
        Args:
            graph: The graph to assign to the embedding.
            as_ref: Whether to assign the graph as a reference or as a copy.
        """
        
        if as_ref:
            self.graph = graph
        else:
            self.graph = graph.clone()
        self.graph_is_ref = as_ref
    
    def save_coordinates(self, c:th.Tensor, name:str):
        """ Save the coordinates to the embedding.
        
        Args:
            c: The coordinates to save.
            name: The name of the coordinates.
        """
        # create embedding dict if it does not exist
        if not hasattr(self, 'embeddings'):
            setattr(self, 'embeddings', {})
        
        self.embeddings[name] = c
        
    def save_coordinates_to_graph(self, c:th.Tensor, name:str):
        """ Save the coordinates to the graph.
        
        Args:
            c: The coordinates to save.
            name: The name of the coordinates.
        """
        assert self.graph is not None, "No graph assigned to embedding. Please assign a graph to the embedding using the assign_graph() method."
        # if entry 'embeddings' does not exist, create it
        if not hasattr(self.graph, 'embeddings'):
            setattr(self.graph, 'embeddings', {})

        self.graph.embeddings[name] = c