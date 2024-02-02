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
import igraph as ig

from causalregnet import utils

class CausalRegNet(object):
    
    def __init__(self,
                 nnodes,
                 mu,
                 theta,
                 W,
                 alpha,
                 beta,
                 agg_type='linear'):

        assert(type(nnodes) == int)

        self.nnodes = nnodes
        self.agg_type = agg_type
        self.mu = mu
        self.theta = theta
        self.W = W
        self.alpha = alpha
        self.beta = beta
        self.gamma = None
        self.b = None

        self.var = self.mu * (1 + self.mu / self.theta)
        
    def check_len(self, vec):
        assert(len(vec) == self.nnodes)
    
    def check_W(self):
        """ Checks that weighted adjacency matrix is (1) acyclic and (2) square. 
        """
        assert(self.W.shape[0] == self.W.shape[1])
        assert(utils.is_dag(self.W))

    def check_alpha(self):
        self.check_len(self.alpha)
        assert(all([a > 1 for a in self.alpha]))

    def check_beta(self):
        self.check_len(self.beta)
        assert(all([0 < b < 1 for b in self.beta]))

    def check_mu(self):
        self.check_len(self.mu)
        assert(all([isinstance(float(m), float) for m in self.mu]))

    def check_theta(self):
        self.check_len(self.theta)
        assert(all([isinstance(float(t), float) for t in self.theta]))

    def check_params(self):
        self.check_W()
        self.check_alpha()
        self.check_beta()
        self.check_mu()
        self.check_theta()

    def check_calib(self):
        assert(self.gamma != None)
        assert(self.b != None)

    def calibrate_sigmoid(self):
        """Calibrates the sigmoid function that models regulatory effect between nodes.

        Returns:
            gamma (list): [d] list of calibrated 'gamma' parameters
            b (list): [d] list of calibrated 'b' parameters
        """
        self.check_params()
        
        gamma = []
        b = []

        if self.agg_type == 'linear':
            
            for j in range(self.nnodes):
                w_sum = np.sum(self.W[:,j])

                if w_sum == 0:
                    gamma_j = 0
                    b_j = 0
                else: 
                    gamma_j = 1 / np.sum(self.W[:,j]) * (np.log(self.alpha[j] / self.beta[j] - 1) - np.log(self.alpha[j] - 1))
                    b_j = - 1 / gamma_j * (np.log(self.alpha[j] - 1) + gamma_j * np.sum(self.W[:,j]))

                # b_j = -1 * np.sum(self.W[:,j])
                # if b_j == 0:
                #     # case: root or master regulator node
                #     # otherwise, cases where the weights sum to 0, under this setting the observational mean will be at regulatory effect 0
                #     gamma_j = 1
                # else: 
                #     # c
                #     gamma_j 


                gamma.append(gamma_j)
                b.append(b_j)
        
        elif self.agg_type == 'linear-znorm':

            for j in range(self.nnodes):

                w_sum_j = 0

                for i in range(self.nnodes):
                    w_sum_j = w_sum_j - self.mu[i] / self.var[i] * self.W[i,j]

                gamma_j = - 1 / w_sum_j * (np.log(self.alpha[j] / self.beta[j] - 1) - np.log(self.alpha[j] - 1))
                b_j = - 1 / gamma_j * np.log(self.alpha[j] - 1)

                gamma.append(gamma_j)
                b.append(b_j)
        else:
            raise ValueError('Invalid aggregation type selected. Please choose from: linear')

        self.gamma = gamma
        self.b = b

    def reg_sigmoid(self, x, j):
        """ Runs the regulatory effect (sigmoid) function given hyperparameters.

        Args:
            x (list) : 
            j (int) :
        """
        self.check_params()

        return(self.alpha[j] * 1 / (1 + np.exp(- self.gamma[j] * (1 * x + self.b[j]))))
    
    def simulate(self, n_samp=1000, intervention_val=None, intervention_type=None):
        """Simulate scRNA-seq dynamics either in observational case (when intervention_set=None) or interventional case (when intervention_set=[target_j]).

        Args:
            n_samp (int): number of samples to be generated
            intervention_set (list): For perfect interventions, this is a list of 'a' values s.t. value j implies X_j = a_j. For stochastic interventions, 
                                    this is a list of values that determines the effect size of the intervention s.t. X_j = x_j * effect_j. 
                                    (negative values imply that this node is skipped)
            intervention_type (str): perfect or stochastic

        """

        self.check_calib()

        def _single_step(X, par_j, j):
            if self.agg_type in ['linear', 'linear-znorm']:
                r_sum = np.zeros(shape=n_samp)
                
                if len(par_j) > 0:
                    if self.agg_type == 'linear':
                        for i in par_j:
                            r_sum += self.W[i, j] * X[:,i] / self.mu[i]
                    elif self.agg_type == 'linear-znorm':
                        for i in par_j:
                            r_sum += self.W[i, j] * (X[:,i] - self.mu[i]) / self.var[i]

                    reg_effect = self.reg_sigmoid(x=r_sum, j=j)

                    if intervention_type == 'stochastic' and intervention_val[j] >= 0:
                        curr_mean = self.mu[j] * reg_effect * intervention_val[j]
                    else:
                        curr_mean = self.mu[j] * reg_effect 

                # root nodes have no regulatory effect
                else:
                    curr_mean = np.repeat(self.mu[j], n_samp) 

                curr_var = curr_mean * (1 + curr_mean / self.theta[j])

                # converting to alternative negative binomial parameters

                curr_p = curr_mean / curr_var
                curr_n = curr_mean ** 2 / (curr_var - curr_mean)
                for k in range(curr_var.shape[0]):
                    if curr_var[k] == 0:
                        curr_p[k] = 1

                if intervention_type == 'deterministic' and intervention_val[j] >= 0:
                    expr = np.repeat(intervention_val[j], n_samp)
                else:
                    expr = np.random.negative_binomial(n=curr_n, p=curr_p, size=n_samp)
                return expr

            else:
                raise ValueError("Unknown aggregation type. Please select from: linear, linear-znorm")
            
        g = ig.Graph.Weighted_Adjacency(self.W.tolist())
        ordered_vertices = g.topological_sorting()

        X = np.zeros([n_samp, self.nnodes])
        for j in ordered_vertices:
            parents_j = g.neighbors(j, mode=ig.IN)
            X[:,j] = _single_step(X=X, par_j=parents_j, j=j)

        return X