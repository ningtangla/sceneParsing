# -*- coding: utf-8 -*-
"""
Created on Mon Sep 04 18:17:59 2017

@author: Edward Coen
"""

import pymc as pm
import numpy as np
import scipy.special as ss

"""
dd as dirichlet_distribution
dp as dirichilet_processs
"""
   
DD_PRIOR_MEAN = 10.
DD_PRIOR_STD = 1.0
DD_PRIOR_CORRELATION = 0.5

MAX_DIMENSION = 6

dp_prior = pm.Lognormal('dp_prior_mean', mu = 0, tau = 4)
dd_prior_mean = pm.Lognormal('dd_prior_mean', mu = 3, tau = 4)

dd_prior = {}
m_list = {}
cov_list = {}
for i in range(2, MAX_DIMENSION + 1):
    m_list[i] = np.array([dd_prior_mean] * i)
    cov_list[i] = np.array([[DD_PRIOR_STD * DD_PRIOR_STD * DD_PRIOR_CORRELATION] * i] * i)
    for j in range(i):
        cov_list[i][j][j] = DD_PRIOR_STD *DD_PRIOR_STD
                
                
@pm.stochastic
def dirichlet_distribution_prior2(value = np.array([DD_PRIOR_MEAN] * 2)):
    if np.any(value <= 0):
        return -np.inf     
    return pm.distributions.mv_normal_cov_like(value, mu = m_list[2], C = cov_list[2])

dd_prior[2] = dirichlet_distribution_prior2

@pm.stochastic
def dirichlet_distribution_prior3(value = np.array([DD_PRIOR_MEAN] * 3)):
    if np.any(value <= 0):
        return -np.inf     
    return pm.distributions.mv_normal_cov_like(value, mu = m_list[3], C = cov_list[3])

dd_prior[3] = dirichlet_distribution_prior3

@pm.stochastic
def dirichlet_distribution_prior4(value = np.array([DD_PRIOR_MEAN] * 4)):
    if np.any(value <= 0):
        return -np.inf     
    return pm.distributions.mv_normal_cov_like(value, mu = m_list[4], C = cov_list[4])

dd_prior[4] = dirichlet_distribution_prior4

@pm.stochastic
def dirichlet_distribution_prior5(value = np.array([DD_PRIOR_MEAN] * 5)):
    if np.any(value <= 0):
        return -np.inf     
    return pm.distributions.mv_normal_cov_like(value, mu = m_list[5], C = cov_list[5])

dd_prior[5] = dirichlet_distribution_prior5
        
@pm.stochastic
def dirichlet_distribution_prior6(value = np.array([DD_PRIOR_MEAN] * 6)):
    if np.any(value <= 0):
        return -np.inf     
    return pm.distributions.mv_normal_cov_like(value, mu = m_list[6], C = cov_list[6])

dd_prior[6] = dirichlet_distribution_prior6
        
@pm.stochastic
def tree_data(value = np.array([[1,2,3,0.4,0.4,0.2], [1,2,3,0.4,0.4,0.2]]), obsevred = True):
    g = value[3:-1]
    print g
    return pm.distributions.dirichlet_like(g, theta = dd_prior[2])

model = pm.Model([dd_prior_mean, dirichlet_distribution_prior2, dirichlet_distribution_prior3, dirichlet_distribution_prior4, dirichlet_distribution_prior5, dirichlet_distribution_prior6, data_tree])
mcmc = pm.MCMC(model)
mcmc.sample(100000, 50000, 1)
pm.Matplot.plot(mcmc)