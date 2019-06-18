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
             
proportion_array = np.array([0.8])
@pm.stochastic
def dirichlet_prior(value = np.array([PRIOR_MEAN] * N_DIMENSION)):
    if np.any(value <= 0):
        return -np.inf
    print value
    return pm.distributions.mv_normal_like(value, mu = np.array([PRIOR_MEAN] * N_DIMENSION), tau = PRIOR_COV)

proportion = pm.Dirichlet('propotion', theta = dirichlet_prior, value = proportion_array, observed=True)
model = pm.Model([proportion, dirichlet_prior])
#mcmc1 = pm.MCMC(model)
#mcmc1.sample(100, 0, 1)
#pm.Matplot.plot(mcmc1)
p = pm.MAP(model)
p.fit()
print 'ok'
print dirichlet_prior.value