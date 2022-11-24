import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import torch as th
import torch_geometric as pyg

from .distances import *
from .graph_metrics import *

from .utils import get_ram
from .Logger import print

def pearson_corr(pos_1:th.Tensor, pos_2:th.Tensor):
    """ Calculates the Pearson correlation between two sets of positions. The sets are given as [N,d] tensors.
        N is the number of data points, and d is the number of dimensions. 

    Args:
        pos_1 (th.Tensor): A [N,d] tensor of positions.
        pos_2 (th.Tensor): A [N,d] tensor of positions.
    Returns:
        float: The Pearson correlation coefficient.
    """

    dist_func = hyperbolic_distance_matrix

    # calculate the distance matrices
    dist_mat1 = dist_func(pos_1, pos_1)
    dist_mat2 = dist_func(pos_2, pos_2)
    
    # replace all nans with 0
    dist_mat1 = th.nan_to_num_(dist_mat1, nan=0)
    dist_mat2 = th.nan_to_num_(dist_mat2, nan=0)

    # calculate the pearson correlation
    corr = np.corrcoef(dist_mat1.detach().numpy(), dist_mat2.detach().numpy())[0,1]

    return corr, dist_mat1, dist_mat2