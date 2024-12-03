"""
Copyright (c) 2024 XX

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
                 agg_type='linear',
                 reg_constant=None):

        assert(isinstance(nnodes, int))

        self.nnodes = nnodes
        self.agg_type = agg_type
        self.mu = mu
        self.theta = theta
        self.W = W
        self.alpha = alpha
        self.beta = beta
        self.reg_constant = reg_constant

        self.gamma = None
        self.b = None

        self.var = self.mu * (1 + self.mu / self.theta)

    def _check_len(self, vec):
        assert(len(vec) == self.nnodes)

    def _check_W(self):
        """ Checks that weighted adjacency matrix is (1) acyclic and (2) square.
        """
        assert(self.W.shape[0] == self.W.shape[1])
        assert(utils.is_dag(self.W))

    def _check_alpha(self):
        self._check_len(self.alpha)
        assert(all([a > 1 for a in self.alpha]))

    def _check_beta(self):
        self._check_len(self.beta)
        assert(all([0 < b < 1 for b in self.beta]))

    def _check_mu(self):
        self._check_len(self.mu)
        assert(all([isinstance(float(m), float) for m in self.mu]))

    def _check_theta(self):
        self._check_len(self.theta)
        assert(all([isinstance(float(t), float) for t in self.theta]))

    def _check_params(self):
        self._check_W()
        self._check_alpha()
        self._check_beta()
        self._check_mu()
        self._check_theta()

    def _check_calib(self):
        assert(self.gamma is not None)
        assert(self.b is not None)

    def calibrate_sigmoid(self):
        """Calibrates the sigmoid function that models regulatory effect between nodes.

        Returns:
            gamma (list): [d] list of calibrated 'gamma' parameters
            b (list): [d] list of calibrated 'b' parameters
        """
        self._check_params()

        gamma = []
        b = []

        if self.agg_type == 'linear' and self.reg_constant is None:

            for j in range(self.nnodes):
                w_sum = np.sum(self.W[:,j])

                if w_sum == 0:
                    gamma_j = 0
                    b_j = 0
                else:
                    b_j = np.log(self.alpha[j]/self.beta[j] - 1) * np.sum(self.W[:,j]) / (np.log(self.alpha[j] - 1) - np.log(self.alpha[j] / self.beta[j] - 1))
                    gamma_j = - 1 / b_j * np.log(self.alpha[j] / self.beta[j] - 1)

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
        elif self.agg_type == 'linear':

            for j in range(self.nnodes):
                w_sum = np.sum(self.W[:,j])

                if w_sum == 0:
                    gamma_j = 0
                    b_j = 0

                else:
                    b_j = np.log(self.alpha[j]/self.beta[j] - 1) * np.sum(self.W[:,j]) / (np.log(self.alpha[j] - 1) - np.log(self.alpha[j] / self.beta[j] - 1))
                    b_j = b_j - self.reg_constant[j]
                    gamma_j = - 1 / b_j * np.log(self.alpha[j] / self.beta[j] - 1)


                b.append(b_j)
                gamma.append(gamma_j)

        else:
            raise ValueError('Invalid aggregation type selected. Please choose from: linear')

        self.gamma = gamma
        self.b = b

    def calibrate_reg_constant(self, q=0.75):
        """ Calibrates the regulatory constant that ensures the parents of each gene don't have overly great effects and are fully expressed
        """

        new_reg_constant: list[float] = []

        x_temp = self.simulate()
        x_max = x_temp.max(axis=0)

        for j in range(self.nnodes):
            w_sum_j = np.quantile(self.W[:,j] * x_max / self.mu, q=q)
            new_reg_constant.append(w_sum_j)

        self.reg_constant = new_reg_constant

    def reg_sigmoid(self, x, j):
        """ Runs the regulatory effect (sigmoid) function given hyperparameters.

        Args:
            x (list) : data to be passed through sigmoid function
            j (int) : node for which the sigmoid function is being run
        """
        self.check_params()
        try:
            val = np.array(self.alpha[j] * 1 / (1 + np.exp(- self.gamma[j] * (1 * x + self.b[j]))), dtype=np.double)
        except ValueError:
            pass

        return(val)

    def simulate(self, n_samp=1000, intervention_val=None, intervention_type=None):
        """Simulate scRNA-seq dynamics either in observational case (when intervention_set=None) or interventional case (when intervention_set=[target_j]).

        Args:
            n_samp (int): number of samples to be generated
            intervention_set (list): For perfect interventions, this is a list of 'a' values s.t. value j implies X_j = a_j. For stochastic interventions,
                                    this is a list of values that determines the effect size of the intervention s.t. X_j = x_j * effect_j.
                                    (negative values imply that this node is skipped / not intervened upon)
            intervention_type (str): perfect or stochastic

        """

        self.check_calib()

        def _single_step(X, par_j, j):

            if self.agg_type in ['linear', 'linear-znorm']:
                r_sum = np.zeros(shape=n_samp, dtype=np.double)

                if len(par_j) > 0:
                    if self.agg_type == 'linear':
                        for i in par_j:
                            r_sum += self.W[i, j] * (X[:,i] / self.mu[i])
                        if self.reg_constant is not None:
                            r_sum += self.reg_constant[j]
                    elif self.agg_type == 'linear-znorm':
                        for i in par_j:
                            r_sum += self.W[i, j] * (X[:,i] - self.mu[i]) / self.var[i]

                    reg_effect = self.reg_sigmoid(x=r_sum, j=j)

                    if intervention_type == 'stochastic' and intervention_val[j] >= 0:
                        curr_mean = self.mu[j] * reg_effect * intervention_val[j]
                    elif intervention_type == 'deterministic' and intervention_val[j] >= 0:
                        curr_mean = np.repeat(intervention_val[j], n_samp)
                    else:
                        curr_mean = self.mu[j] * reg_effect

                # root nodes have no regulatory effect
                else:
                    curr_mean = np.repeat(self.mu[j], n_samp)

                curr_var = curr_mean * (1 + curr_mean / self.theta[j])

                # converting to alternative negative binomial parameters

                curr_p = np.zeros([n_samp])
                curr_n = np.zeros([n_samp])

                for c in range(n_samp):
                    if curr_var[c] == 0:
                        curr_p[c] = 1
                    else:
                        curr_p[c] = curr_mean[c] / curr_var[c]

                    if curr_mean[c] == curr_var[c]:
                        curr_n[c] = np.inf
                    else:
                        curr_n[c] = curr_mean[c] ** 2 / (curr_var[c] - curr_mean[c])

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


    def simulate_meanwise(self, n_samp=1000, intervention_val=None, intervention_type=None):
        """Simulate scRNA-seq dynamics either in observational case (when intervention_set=None) or interventional case (when intervention_set=[target_j]).

        Args:
            n_samp (int): number of samples to be generated
            intervention_set (list): For perfect interventions, this is a list of 'a' values s.t. value j implies X_j = a_j. For stochastic interventions,
                                    this is a list of values that determines the effect size of the intervention s.t. X_j = x_j * effect_j.
                                    (negative values imply that this node is skipped)
            intervention_type (str): perfect or stochastic

        """

        self.check_calib()

        def _single_step(X, mu_R, par_j, j):
            if self.agg_type in ['linear', 'linear-znorm']:
                r_sum = 0

                if len(par_j) > 0:
                    if self.agg_type == 'linear':
                        for i in par_j:
                            r_sum += self.W[i, j] * mu_R[i] / self.mu[i]
                    elif self.agg_type == 'linear-znorm':
                        for i in par_j:
                            r_sum += self.W[i, j] * (mu_R[i] - self.mu[i]) / self.var[i]

                    reg_effect = self.reg_sigmoid(x=r_sum, j=j)

                    curr_mean = self.mu[j] * reg_effect

                # root nodes have no regulatory effect
                else:
                    curr_mean = self.mu[j]

                if intervention_val is not None and intervention_val[j] >= 0:
                    if intervention_type == 'deterministic':
                        curr_mean = intervention_val[j]
                    else:
                        curr_mean = curr_mean * intervention_val[j]

                curr_var = curr_mean * (1 + curr_mean / self.theta[j])

                # converting to alternative negative binomial parameters

                ### TODO: we can't have intervention_val = 0 since then our p and n are infinity

                if curr_mean == 0:
                    temp_mean = 1e-6
                    curr_p = temp_mean / curr_var
                    curr_n = temp_mean ** 2 / (curr_var - temp_mean)
                else:
                    curr_p = curr_mean / curr_var
                    curr_n = curr_mean ** 2 / (curr_var - curr_mean)

                if intervention_type == 'deterministic' and intervention_val[j] >= 0:
                    expr = np.repeat(intervention_val[j], n_samp)
                else:
                    expr = np.random.negative_binomial(n=curr_n, p=curr_p, size=n_samp)
                return expr, curr_mean

            else:
                raise ValueError("Unknown aggregation type. Please select from: linear, linear-znorm")

        g = ig.Graph.Weighted_Adjacency(self.W.tolist())
        ordered_vertices = g.topological_sorting()

        X = np.zeros([n_samp, self.nnodes])
        mu_R = self.mu.copy()
        for j in ordered_vertices:
            parents_j = g.neighbors(j, mode=ig.IN)
            X[:,j], mu_R[j] = _single_step(X=X, mu_R=mu_R, par_j=parents_j, j=j)

        return X
