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
import networkx as nx

import utils

class PerturbSCM(object):
    
    def __init__(self,
                 nnodes):
        self.nnodes = nnodes
        self.adj_mat = None

        
    def set_graph(self, adj_mat):
        """ Checks that causal graph is (1) acyclic and (2) square. 
        """
        assert(adj_mat.shape[0] == adj_mat.shape[1])
        assert(utils.is_dag(adj_mat))

        self.adj_mat = adj_mat

    def calibrate_sigmoid(self, W, alpha, beta, type='linear'):
        """Calibrates the sigmoid function that models regulatory effect between nodes.

        Args:
            W (np.ndarray): [d, d] weighted adj matrix of DAG
            alpha (int): [d] list of desired maximum regulatory effect of parents
            beta (int): [d] list of desired minimum regulatory effect of parents
            type (str): mean-norm, z-norm, single-mean

        Returns:
            gamma (list): [d] list of calibrated 'gamma' parameters
            b (list): [d] list of calibrated 'b' parameters
        """

        n = W.shape[0]
        
        gamma = []
        b = []

        if type == 'linear':
            
            for j in range(n):
                gamma_j = np.log(alpha[j] / beta[j] - 1) - np.log(alpha[j] - 1)
                b_j = - 1 / gamma_j * (np.log(alpha[j] - 1) + gamma_j * np.sum(W[:,j]))

                gamma.append(gamma_j)
                b.append(b_j)

        else:
            ValueError('Invalid aggregation type selected. Please choose from: linear')

        return gamma, b

    def c_sigmoid(self, x, alpha_j, gamma_j, b_j):
        """ Runs the regulatory effect (sigmoid) function given hyperparameters.

        Args:
            x (list) : 
            alpha_j (float) : 
            gamma_j (float) : 
            b_j (float) : 

        """
        return(alpha_j * 1 / (1 + np.exp(- gamma_j * (1 * x + b_j))))