"""
Copyright (c) 2024 Luka Kovacevic

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np
import pandas as pd
import igraph as ig
import random

from scipy.special import expit as sigmoid

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def is_dag(W):
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()

def generate_dag(d, m, graph_type='ER'):
    """Simulate random DAG with d nodes and m edges.

    Args:
        d (int): num of nodes
        m (int): expected num of edges
        graph_type (str): ER, SF, BP

    Returns:
        B (np.ndarray): [d, d] binary adj matrix of DAG
    """

    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0])) 
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        return np.tril(_random_permutation(B_und), k=-1)

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)
    
    if graph_type == 'ER':
        # Erdos-Renyi
        G_und = ig.Graph.Erdos_Renyi(n=d, m=m)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == 'ER-p':
        G_und = ig.Graph.Erdos_Renyi(n=d, p=m)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == 'SF':
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(n=d, m=int(round(m / d)), directed=True)
        B = _graph_to_adjmat(G)
    elif graph_type == 'BP':
        # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
        top = int(0.2 * d)
        G = ig.Graph.Random_Bipartite(top, d - top, m=m, directed=True, neimode=ig.OUT)
        B = _graph_to_adjmat(G)
    else:
        raise ValueError('unknown graph type')
    B_perm = _random_permutation(B)
    assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
    return B_perm

def assign_parent_pairs(B):
    """Given an adjacency matrix B, this method pairs parents of nodes up for balanced specification of W.

        Args:
            B (np.ndarray): [d, d] adjacency matrix

        Returns:
            P (np.ndarray): [d, d] pair indexed matrix
    """
    P = np.zeros(B.shape)
    count = 1

    for j in range(B.shape[1]):
        parent_list = np.where(B[:,j] == 1)[0].tolist()

        parent1 = None

        while len(parent_list) > 0:
            idx = random.randrange(0, len(parent_list))
            new_parent = parent_list.pop(idx)
            if parent1 is None:
                if len(parent_list) > 0:
                    parent1 = new_parent
                else:
                    break
            else:
                P[parent1, j] = count 
                P[new_parent, j] = count
                count += 1
    return P

def generate_W(B, w_ranges=((-2.0, -0.5), (0.5, 2.0)), balanced=False, balancing_noise_std=0.1):
    """Simulate weight edges given a binary DAG adjacency matrix.

    Args:
        B (np.ndarray): [d, d] binary adj matrix of DAG
        w_ranges (tuple): disjoint weight ranges
        balanced (boolean): determines whether parents weights are balanced to ensure consistency under linear aggregation
        balancing_norm_scale (float): standard deviation used to generate noise in paired weights (only works when `balanced = True`)

    Returns:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
    """
    W = np.zeros(B.shape)
    S = np.random.randint(len(w_ranges), size=B.shape)

    # iterate through each group determined by S (i.e. which w_range each entry belongs to)
    for i, (low, high) in enumerate(w_ranges):
        U = np.random.uniform(low=low, high=high, size=B.shape)
        W += B * (S == i) * U


    if balanced:
        P = assign_parent_pairs(B)

        for k in set(P.flatten()):
            if k != 0: # 0 indicates unpaired entries
                matching_idx = np.where(P == k)
                W[matching_idx[0][1], matching_idx[1][1]] = -1 * W[matching_idx[0][0], matching_idx[1][0]] + np.random.normal(loc=0, scale=balancing_noise_std, size=1)

    return W

def is_dag(W):
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()

def generate_node_params(nnodes,
                    min_mu=5, 
                    max_mu=30, 
                    min_theta=2,
                    max_theta=15,
                    min_alpha=2, 
                    max_alpha=2,
                    min_beta=0.1, 
                    max_beta=0.1):

    mu=np.random.uniform(min_mu, max_mu, nnodes)
    theta=np.random.uniform(min_theta, max_theta, nnodes)
    alpha=np.random.uniform(min_alpha, max_alpha, nnodes)
    beta=np.random.uniform(min_beta, max_beta, nnodes)

    return mu, theta, alpha, beta