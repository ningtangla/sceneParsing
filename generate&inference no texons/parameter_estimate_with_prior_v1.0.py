# -*- coding: utf-8 -*-
"""
Created on Mon Sep 04 18:17:59 2017

@author: Edward Coen
"""

import pymc as pm
import numpy as np
import scipy.special as ss

PRIOR_STD = 1
PRIOR_CORRELATION = 0.5

MAX_DIMENSION = 6

dirichilet_process_prior = pm.Lognormal('dirichilet_process_prior_mean', mu = 0, tau = 1)
dirichilet_distribution_prior_mean = pm.Lognormal('dirichilet_distribution_prior_mean', mu = 3, tau = 1.0/4)

a = pm.Beta('aa', alpha = dirichilet_process_prior, beta = dirichilet_distribution_prior_mean)
mcmc2 = pm.MCMC([dirichilet_process_prior, dirichilet_distribution_prior_mean, a])
mcmc2.sample(1000,0,1)
pm.Matplot.plot(mcmc2)
dirichilet_distribution_prior_std = pm.Uniform('dirichilet_distribution_prior_std', lower = 0.99, upper = 1.001)
dirichilet_distribution_prior_correlation = pm.Uniform('dirichilet_distribution_prior_correlation', lower = 0.499, upper = 0.501)
dirichilet_distribution_prior = {}
m_list = {}
cov_list = {}
for i in range(1, MAX_DIMENSION + 1):
    print i
    m_list[i] = np.array([dirichilet_distribution_prior_mean] * i)
    cov_list[i] = np.array([[dirichilet_distribution_prior_std * dirichilet_distribution_prior_std * dirichilet_distribution_prior_correlation] * i] * i)
    for j in range(i):
        cov_list[i][j][j] = dirichilet_distribution_prior_std * dirichilet_distribution_prior_std
    dirichilet_distribution_prior[i] = pm.MvNormalCov('dirichilet_distribution_prior', mu = m_list[i], C = cov_list[i])
    
data_array = np.array([[1.0, 2.0, 3.0, 0.4, 0.4, 0.2], [1.0, 1.0, 0.4, 0.6]])
@pm.stochastic(dtype = float)
def data_tree(value = [1.0, 1.0, 0.4, 0.6], observed = True):
    n_dimension = len(value) / 2

#           
#    for j in range(len(permutation_table)):
#        guest_arrange_num = guest_total_num
#        comb_num = 1
#        for jj in range(len(permutation_table[j])):    
#            guest_num_per_table = permutation_table[j][jj]
#            comb_num = comb_num * int(comb(guest_arrange_num - 1, guest_num_per_table - 1))
#            guest_arrange_num = guest_arrange_num - guest_num_per_table
#        comb_total_num = comb_total_num + comb_num        
    return pm.distributions.dirichlet_like(value[n_dimension:], theta = dirichilet_distribution_prior[n_dimension])

model = pm.Model([dirichilet_distribution_prior_mean, dirichilet_distribution_prior_std, dirichilet_distribution_prior_correlation, dirichilet_distribution_prior, data_tree])
mcmc = pm.MCMC(model)
mcmc.sample(100000, 50000, 1)
pm.Matplot.plot(mcmc)