# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 22:46:49 2017

@author: Edward Coen
"""
import pymc as pm
import cv2
import numpy as np
import scipy.stats as st

ITER_NUM = 10
PRIOR_MEAN = 10
PRIOR_STD = 1
PRIOR_CORRELATION = 0.0
N_DIMENSION = 2

prior_mean = pm.Lognormal('prior_mean', mu = 0, tau = 4)
PRIOR_COV = np.array([[PRIOR_STD * PRIOR_STD * PRIOR_CORRELATION] * N_DIMENSION] * N_DIMENSION)
for i in range(N_DIMENSION):
    PRIOR_COV[i][i] = PRIOR_STD * PRIOR_STD
               
proportion_array = np.array([0.9])
for i in range(ITER_NUM):
    @pm.stochastic
    def dirichlet_prior(value = np.array([PRIOR_MEAN] * N_DIMENSION)):
        if np.any(value <= 0):
            return -np.inf
        return pm.distributions.mv_normal_cov_like(value, mu = np.array([prior_mean] * N_DIMENSION), C = PRIOR_COV)
    
#    @pm.stochastic
#    def proportion(value = proportion_array, observed = True):
#        p = 1.0
#        for j in range(len(proportion_array)):
#            print proportion_array[j]
#            print dirichlet_prior.value
#            p = pm.distributions.dirichlet_like(x = value[j], theta = dirichlet_prior)
#        return p 
    
    proportion = pm.Dirichlet('propotion', theta = dirichlet_prior, value = proportion_array, observed=True)
    model = pm.Model([proportion, dirichlet_prior, prior_mean])
    p = pm.MAP(model)
    p.fit()
    proportion_array = dirichlet_prior.value[:-1] / dirichlet_prior.value.sum()
    print proportion_array
    temp = prior_mean.value
#    print temp, dirichlet_prior.value