# -*- coding: utf-8 -*-
"""
Created on Tue Nov 07 16:47:56 2017

@author: Edward Coen
"""

import pymc as pm
import numpy as np
import scipy.special as ss

DATA_SIZE = 2
ORIGIN = np.array([6, 0, 0, 0, 0, 0, 0, 0, 0])

### hyperprior-prior
@pm.stochastic
def prior_for_beta_distribution(num_partition = 2, value = [1.] * 2):
    if np.any(value <= 0):
        return -np.inf     
    return (-0.5 - num_partition) * np.log(np.array(value).sum())

@pm.stochastic
def prior_for_dirichlet_distribution_partition3(num_partition = 3, value = [1.] * 3):
    if np.any(value <= 0):
        return -np.inf     
    return (-0.5 - num_partition) * np.log(np.array(value).sum())

@pm.stochastic
def prior_for_dirichlet_distribution_partition4(num_partition = 4, value = [1.] * 4):
    if np.any(value <= 0):
        return -np.inf     
    return (-0.5 - num_partition) * np.log(np.array(value).sum())

@pm.stochastic
def prior_for_dirichlet_distribution_partition5(num_partition = 5, value = [1.] * 5):
    if np.any(value <= 0):
        return -np.inf     
    return (-0.5 - num_partition) * np.log(np.array(value).sum())

@pm.stochastic
def prior_for_dirichlet_distribution_partition6(num_partition = 6, value = [1.] * 6):
    if np.any(value <= 0):
        return -np.inf     
    return (-0.5 - num_partition) * np.log(np.array(value).sum())

dirichlet_process_prior = pm.Lognormal('dirichlet_process_prior', mu = 0, tau = 4)

@pm.stochastic(dtype = int)
def cut_1(value = np.array([[1, 5, 0, 0, 0, 0, 0, 2, 1]] * DATA_SIZE):  
    p_all = 0
    for i in range(DATA_SIZE):
        parent_node_index = value[i][7]
        cut_partition_num = value[i][8]
        rest_partion_num = 6 - parent_node_index - cut_partition_num
        rest_index_start = parent_node_inde + cut_partition_num
        next_parent_node_index = range(parent_node_index, rest_index_start)[value[i][9]]
        
        
        if (value[i][7] ==  np.min(np.where(ORIGIN > 1))) \
            and (value[i][next_parent_node_index] == np.min(np.where(value[i] > 1))) \
            and (list(value[i][next_parent_node_index + 1 : next_parent_node_index + 1 + rest_partion_num]) == list(ORIGIN[parent_node_index + 1:])
        
        else:
            return -np.inf
#        if np.sum(value[i][0:-2]) != ORIGIN[0]:
#            return -np.inf
#        if value[i][7] != ORIGIN:
#            
#        if
    
    return p_all

def CRP_level_2

@pm.observed(dtype = int)
def test(value = [[1, 3], [5, 7]]):
    p = 1
    for i in range(len(value)):
        p = p * ss.gamma(prior_for_beta_distribution[0].value) * np.sum(value[i]) 
    return p

model = pm.Model([prior_for_beta_distribution, test])
mcmc = pm.MCMC(model)
mcmc.sample(100000, 50000, 1)
pm.Matplot.plot(mcmc)   
#print prior_for_beta_distribution[0].value     