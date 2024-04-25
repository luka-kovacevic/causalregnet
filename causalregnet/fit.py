from scipy.optimize import minimize
from scipy.special import gammaln
import numpy as np
import pandas as pd
import math

def sigmoid(x):
    return(1 / (1 + np.exp(-x)))

def nb_pmf(x, mu, theta):
    return(np.exp(gammaln(x + theta) - gammaln(x + 1) - gammaln(theta) + theta * np.log(theta / (theta + mu)) + x * np.log(mu / (theta + mu))))

def zinb_pmf(x, pi, mu, theta):
    is_zero = (x == 0)
    pmf_zeros = pi + (1 - pi) * nb_pmf(x=0, mu=mu, theta=theta)
    pmf_count = (1 - pi) * nb_pmf(x=x, mu=mu, theta=theta)
    return np.where(is_zero, pmf_zeros, pmf_count)

def nb_loss(params, x):
    mu, theta = params
    loss = nb_pmf(x=x, mu=mu, theta=theta)
    return(-1 * np.sum(np.log(loss)))

def zinb_loss(params, x):
    pi, mu, theta = params

    loss = zinb_pmf(x=x, pi=pi, mu=mu, theta=theta)

    return (-1 * np.sum(np.log(loss)))

def fit_nb(data):
    mu = np.mean(data)
    variance = np.var(data)

    theta = (mu ** 2) / (variance - mu)

    initial_guess = [mu, theta]

    # Minimize the negative log-likelihood
    result = minimize(nb_loss, 
                      initial_guess, 
                      args=(data,),
                      bounds=[(1e-9, None), (1e-9, None)],
                      method="L-BFGS-B")  # n > 0

    # Store the fitted parameters
    mu, theta = result.x
    return mu, theta

def fit_zinb(data):
    # mu = np.mean(data)
    # variance = np.var(data)

    # theta = (mu ** 2) / (variance - mu)

    # pi = 0.2

    # initial_guess = [pi, mu, theta]

    # result = minimize(zinb_loss, initial_guess, args=(data,),
    #                   bounds=[(0, 1), (0.01, None), (None, None)])
    
    initial_guess = [1, 1, 0.5]  # pi, r, p
    bounds = [(0+1e-9, 1 - 1e-9), (1e-9, None), (1e-9, None)]  # Ensure parameters are within valid ranges
    result = minimize(zinb_loss, 
                      initial_guess, 
                      args=(data,), 
                      bounds=bounds, 
                      method="L-BFGS-B")

    pi, mu, theta = result.x
    return pi, mu, theta