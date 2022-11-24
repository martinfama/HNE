import os, sys

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

import LinkPrediction as LP

def MAP(G:pyg.data.Data, pos_name=''):
    return