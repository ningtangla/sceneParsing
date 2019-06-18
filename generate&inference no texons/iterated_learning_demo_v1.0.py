# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 22:46:49 2017

@author: Edward Coen
"""
import pymc as pm
import cv2
import numpy as np
import scipy.stats as st

PRIOR_MEAN = 10.
PRIOR_STD = 1.
PRIOR_CORRELATION = 0.9
N_DIMENSION = 2

PRIOR_COV = np.array([[PRIOR_STD * PRIOR_STD * PRIOR_CORRELATION] * N_DIMENSION] * N_DIMENSION)
for i in range(N_DIMENSION):
    PRIOR_COV[i][i] = PRIOR_STD * PRIOR_STD
             
@pm.stochastic
def dirichlet_prior(value = np.array([15., 5.])):
    if np.any(value <= 0):
        return -np.inf
    print value
    return pm.distributions.mv_normal_like(value, mu = np.array([PRIOR_MEAN] * N_DIMENSION), tau = PRIOR_COV)

proportion = pm.Dirichlet('propotion', theta = dirichlet_prior)
mcmc1 = pm.MCMC([proportion, dirichlet_prior])
mcmc1.sample(100, 0, 1)
pm.Matplot.plot(mcmc1)