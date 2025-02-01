import igraph as ig
import numpy as np
from numpy.typing import NDArray


class CausalRegNet(object):
    gamma: NDArray | None
    b: NDArray | None

    def __init__(
        self,
        nnodes: int,
        mu: NDArray,
        theta: NDArray,
        W: NDArray,
        alpha: NDArray,
        beta: NDArray,
        agg_type: str = "linear",
        reg_constant: float | None = None,
    ):
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

        self._calibrate_sigmoid()

    def _calibrate_sigmoid(self):
        """Calibrates the sigmoid function that models regulatory effect between nodes.

        Returns:
            gamma (list): [d] list of calibrated 'gamma' parameters
            b (list): [d] list of calibrated 'b' parameters
        """
        gamma = []
        b = []

        if self.agg_type == "linear" and self.reg_constant is None:
            for j in range(self.nnodes):
                w_sum = np.sum(self.W[:, j])

                if w_sum == 0:
                    gamma_j = 0
                    b_j = 0
                else:
                    b_j = (
                        np.log(self.alpha[j] / self.beta[j] - 1)
                        * np.sum(self.W[:, j])
                        / (np.log(self.alpha[j] - 1) - np.log(self.alpha[j] / self.beta[j] - 1))
                    )
                    gamma_j = -1 / b_j * np.log(self.alpha[j] / self.beta[j] - 1)

                gamma.append(gamma_j)
                b.append(b_j)

        elif self.agg_type == "linear-znorm":
            for j in range(self.nnodes):
                w_sum_j = 0

                for i in range(self.nnodes):
                    w_sum_j = w_sum_j - self.mu[i] / self.var[i] * self.W[i, j]

                gamma_j = -1 / w_sum_j * (np.log(self.alpha[j] / self.beta[j] - 1) - np.log(self.alpha[j] - 1))
                b_j = -1 / gamma_j * np.log(self.alpha[j] - 1)

                gamma.append(gamma_j)
                b.append(b_j)
        elif self.agg_type == "linear":
            for j in range(self.nnodes):
                w_sum = np.sum(self.W[:, j])

                if w_sum == 0:
                    gamma_j = 0
                    b_j = 0

                else:
                    b_j = (
                        np.log(self.alpha[j] / self.beta[j] - 1)
                        * np.sum(self.W[:, j])
                        / (np.log(self.alpha[j] - 1) - np.log(self.alpha[j] / self.beta[j] - 1))
                    )
                    b_j = b_j - self.reg_constant[j]
                    gamma_j = -1 / b_j * np.log(self.alpha[j] / self.beta[j] - 1)

                b.append(b_j)
                gamma.append(gamma_j)

        else:
            raise ValueError("Invalid aggregation type selected. Please choose from: linear")

        self.gamma = np.array(gamma)
        self.b = np.array(b)

    def calibrate_reg_constant(self, q=0.75):
        """Calibrates the regulatory constant."""

        new_reg_constant = []

        x_temp = self.simulate()
        x_max = x_temp.max(axis=0)

        for j in range(self.nnodes):
            w_sum_j = np.quantile(self.W[:, j] * x_max / self.mu, q=q)
            new_reg_constant.append(w_sum_j)

        self.reg_constant = new_reg_constant

    def reg_sigmoid(self, x, j):
        """Runs the regulatory effect (sigmoid) function given hyperparameters.

        Args:
            x (list) : data to be passed through sigmoid function
            j (int) : node for which the sigmoid function is being run
        """
        try:
            val = np.array(
                self.alpha[j] * 1 / (1 + np.exp(-self.gamma[j] * (1 * x + self.b[j]))),
                dtype=np.double,
            )
        except ValueError:
            pass

        return val

    def simulate(self, n_samp=1000, intervention_val=None, intervention_type=None):
        """Simulate scRNA-seq expression.

        Args:
            n_samp (int): number of samples to be generated
            intervention_val (list | None): Either None or list
                of intervention values for each node.
            intervention_type (str): perfect or stochastic

        """

        def _single_step(X, par_j, j):
            if self.agg_type in ["linear", "linear-znorm"]:
                r_sum = np.zeros(shape=n_samp, dtype=np.double)

                if len(par_j) > 0:
                    if self.agg_type == "linear":
                        for i in par_j:
                            r_sum += self.W[i, j] * (X[:, i] / self.mu[i])
                        if self.reg_constant is not None:
                            r_sum += self.reg_constant[j]
                    elif self.agg_type == "linear-znorm":
                        for i in par_j:
                            r_sum += self.W[i, j] * (X[:, i] - self.mu[i]) / self.var[i]

                    reg_effect = self.reg_sigmoid(x=r_sum, j=j)

                    if intervention_type == "stochastic" and intervention_val[j] >= 0:
                        curr_mean = self.mu[j] * reg_effect * intervention_val[j]
                    elif intervention_type == "deterministic" and intervention_val[j] >= 0:
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

                if intervention_type == "deterministic" and intervention_val[j] >= 0:
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
            X[:, j] = _single_step(X=X, par_j=parents_j, j=j)

        return X

    def simulate_meanwise(self, n_samp=1000, intervention_val=None, intervention_type=None):
        """Simulate scRNA-seq expression.

        Args:
            n_samp (int): number of samples to be generated
            intervention_set (list): For perfect interventions, this is a list of 'a' values s.t.
                value j implies X_j = a_j.
                For stochastic interventions, this is a list of values that determines the effect
                size of the intervention s.t. X_j = x_j * effect_j.
                negative values imply that this node is skipped
            intervention_type (str): perfect or stochastic

        """

        def _single_step(X, mu_R, par_j, j):
            if self.agg_type in ["linear", "linear-znorm"]:
                r_sum = 0

                if len(par_j) > 0:
                    if self.agg_type == "linear":
                        for i in par_j:
                            r_sum += self.W[i, j] * mu_R[i] / self.mu[i]
                    elif self.agg_type == "linear-znorm":
                        for i in par_j:
                            r_sum += self.W[i, j] * (mu_R[i] - self.mu[i]) / self.var[i]

                    reg_effect = self.reg_sigmoid(x=r_sum, j=j)

                    curr_mean = self.mu[j] * reg_effect

                # root nodes have no regulatory effect
                else:
                    curr_mean = self.mu[j]

                if (intervention_val is not None) and (intervention_val[j] >= 0):
                    if intervention_type == "deterministic":
                        curr_mean = intervention_val[j]
                    else:
                        curr_mean = curr_mean * intervention_val[j]

                curr_var = curr_mean * (1 + curr_mean / self.theta[j])

                if curr_mean == 0:
                    temp_mean = 1e-6
                    curr_p = temp_mean / curr_var
                    curr_n = temp_mean**2 / (curr_var - temp_mean)
                else:
                    curr_p = curr_mean / curr_var
                    curr_n = curr_mean**2 / (curr_var - curr_mean)

                if intervention_type == "deterministic" and intervention_val[j] >= 0:
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
            X[:, j], mu_R[j] = _single_step(X=X, mu_R=mu_R, par_j=parents_j, j=j)

        return X
