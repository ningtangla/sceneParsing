# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 22:46:49 2017

@author: Edward Coen
"""
import pymc as pm
import cv2
import numpy as np
import scipy.stats as st

ITER_NUM = 1
PRIOR_MEAN = 10.
PRIOR_STD = 1.
PRIOR_CORRELATION = 0.0
N_DIMENSION = 2

prior_mean = pm.Lognormal('prior_mean', mu = 0, tau = 4)
#mcmc1 = pm.MCMC([prior_mean])
#mcmc1.sample(100000, 50000, 1)
#pm.Matplot.plot(mcmc1)
#mcmc1.summary()


PRIOR_COV = np.array([[PRIOR_STD * PRIOR_STD * PRIOR_CORRELATION] * N_DIMENSION] * N_DIMENSION)
for i in range(N_DIMENSION):
    PRIOR_COV[i][i] = PRIOR_STD * PRIOR_STD
             
proportion_array = np.array([[0.8],[0.7]]) 
for i in range(ITER_NUM):
            
    @pm.stochastic
    def dirichlet_prior(value = np.array([PRIOR_MEAN] * N_DIMENSION), observed = False):
        if np.any(value <= 0):
            return -np.inf
        return pm.distributions.mv_normal_cov_like(value, mu = np.array([prior_mean] * N_DIMENSION), C = PRIOR_COV)
    
    proportion = pm.Dirichlet('propotion', theta = dirichlet_prior, value = proportion_array, observed = True)
    model = pm.Model([proportion, dirichlet_prior, prior_mean])
    
    mcmc = pm.MCMC(model)
    mcmc.sample(100000, 50000, 1)
    pm.Matplot.plot(mcmc)
    hyp_theta = np.average(mcmc.trace('dirichlet_prior')[:], axis = 0)
    z = np.average(mcmc.trace('prior_mean')[:], axis = 0)
    print hyp_theta, z
    proportion_array = pm.distributions.dirichlet_expval(hyp_theta)[0]
    print proportion_array