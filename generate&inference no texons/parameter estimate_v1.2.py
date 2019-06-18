# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 19:41:07 2017

@author: Edward Coen
"""
from __future__ import division
import scipy.special
from scipy.misc import comb
import random
import math
import scipy.stats
import numpy as np
import networkx as nx
from networkx.algorithms.traversal.depth_first_search import dfs_successors
import itertools as it
import operator as op
import pandas as pd
import cv2 
import sys 
import itertools 
import matplotlib.pyplot as plt 
import graphviz
from sympy import *

sys.setrecursionlimit(1000000000)

TOLERANCE = 0.0000001
MAX_STEPS_NUM = 1000
ITER_NUM = 50

GRAPH_NUM = 1
GRAPH_STRAT_NUM = 1003

INFER_SAMPLE_NUM_TOTAL = 1

"""
THETA as the GAMMA in generate and inference for nCRP, just differ from the gamma function
THETA range: [THETA_AVERAGE - THETA_INTERVAL * THETA_RANGE_NUM, THETA_AVERAGE + THETA_INTERVAL * THETA_RANGE_NUM]
"""
THETA_RANGE_NUM = 0 
THETA_AVERAGE = 2.0
THETA_INTERVAL = 0.2
MIN_THETA = 0.01
MAX_THETA = 5

"""




same range as THETA
"""
ALPHA_RANGE_NUM = 0
ALPHA_AVERAGE = 20
ALPHA_INTERVAL = 0.5
MIN_ALPHA = 0.1
MAX_ALPHA = 30

def make_target_function(graph_num, theta_range_num, alpha_range_num):
    f_posterior = Function('f_posterior')
    f_normalization = Function('f_normalization')
    f_infer = Function('f_infer')
    f_max_target = 0
    
    for i in range(GRAPH_STRAT_NUM , graph_num + GRAPH_STRAT_NUM ):
        for k in range(-theta_range_num, theta_range_num + 1):
            for j in range(-alpha_range_num, alpha_range_num + 1): 
                g0 = nx.read_gpickle('E:/ncrp_infer/most_like_tree_alpha['+str(ALPHA_AVERAGE + j * ALPHA_INTERVAL)+']_gamma['+str(THETA_AVERAGE + j * THETA_INTERVAL)+']_'+str(i)+'.gpickle')
                
                
                f_posterior_list = []
                possible_trees_num = g0.node[1]['possible_trees_num']
                p_original =  g0.node[1]['p_poster_list']
                f_normalization = 0
                for l in range(possible_trees_num):
                    g = nx.read_gpickle('E:/ncrp_infer/possible_tree_infer_alpha['+str(ALPHA_AVERAGE + j * ALPHA_INTERVAL)+']_gamma['+str(THETA_AVERAGE + j * THETA_INTERVAL)+']_'+'img_'+str(i)+'_tree_'+str(l)+'.gpickle')
                    nonleaf_nodes = [ n for n,d in g.out_degree().items() if d!=0] 
                    
                    all_table_partion_list = map(lambda x: map(lambda y: len(g.node[y]['guest']), nx.DiGraph.successors(g, x)), nonleaf_nodes)
                    all_cut_proportion_list = map(lambda x: g.node[x]['cut_proportion'], nonleaf_nodes)
                    f_posterior = make_posterior_function(all_table_partion_list, all_cut_proportion_list)
                    f_posterior_list.append(f_posterior)
                    f_normalization = f_normalization + f_posterior
#                print p_original   
                p_sample = map(lambda x: x * 1.0 / np.array(p_original).sum(), p_original)
#                ttt = np.array(map(lambda x: pow(x, x), p_sample)).sum()
#                
#                aaa = []
#                bbb = []
                
                theta = Symbol('theta')
                alpha = Symbol('alpha')
                for m in range(possible_trees_num):
                    sample_num = INFER_SAMPLE_NUM_TOTAL * p_sample[m]
#                    print sample_num
                    f_max_target = f_max_target + sample_num * (log(f_posterior_list[m]) - log(f_normalization))
#                    aaa.append(pow(f_posterior_list[m].subs({'theta': 2, 'alpha': 20})/f_normalization.subs({'theta': 2, 'alpha': 20}), p_sample[m]))
#                    bbb.append(f_max_target.subs({'theta': 2, 'alpha': 20}))
#                    print '!!!', (f_posterior_list[m].subs({'theta': 2, 'alpha': 20})/f_normalization.subs({'theta': 2, 'alpha': 20}))
#                z = map(lambda x: x.subs({'theta': 0.967, 'alpha': 21.0})/f_normalization.subs({'theta': 0.967, 'alpha': 21.0}), f_posterior_list)
#                z1 = map(lambda x: x/np.array(z).sum(), z)
#                zzz =  np.array(map(lambda x: pow(z1[x], p_sample[x]), range(len(z1)))).sum()
#                
#                print aaa, bbb
#                print p_sample, np.array(p_sample).sum(), ttt, map(lambda x: pow(x, x), p_sample)
#                print 'ggg'
#                print z1, np.array(z1).sum(), zzz, map(lambda x: pow(z1[x], p_sample[x]), range(len(z1)))
#    print f_max_target.subs({'theta': 2, 'alpha': 20})
#    print f_max_target.subs({'theta': 0.967, 'alpha': 21})
#    print '80!'
#    print f_max_target.subs({'theta': 80, 'alpha': 80})
#    print f_max_target.subs({'theta': 75, 'alpha': 80})
#    print f_max_target.subs({'theta': 70, 'alpha': 80})
#    print f_max_target.subs({'theta': 65, 'alpha': 80})
#    print f_max_target.subs({'theta': 60, 'alpha': 80})
#    print f_max_target.subs({'theta': 55, 'alpha': 80})
#    print f_max_target.subs({'theta': 50, 'alpha': 80})
#    print '75!'
#    print f_max_target.subs({'theta': 80, 'alpha': 75})
#    print f_max_target.subs({'theta': 75, 'alpha': 75})
#    print f_max_target.subs({'theta': 70, 'alpha': 75})
#    print f_max_target.subs({'theta': 65, 'alpha': 75})
#    print f_max_target.subs({'theta': 60, 'alpha': 75})
#    print f_max_target.subs({'theta': 55, 'alpha': 75})
#    print f_max_target.subs({'theta': 50, 'alpha': 75})
#    print '70!'
#    print f_max_target.subs({'theta': 80, 'alpha': 70})
#    print f_max_target.subs({'theta': 75, 'alpha': 70})
#    print f_max_target.subs({'theta': 70, 'alpha': 70})
#    print f_max_target.subs({'theta': 65, 'alpha': 70})
#    print f_max_target.subs({'theta': 60, 'alpha': 70})
#    print f_max_target.subs({'theta': 55, 'alpha': 70})
#    print f_max_target.subs({'theta': 50, 'alpha': 70})
#    print '65!'
#    print f_max_target.subs({'theta': 80, 'alpha': 65})
#    print f_max_target.subs({'theta': 75, 'alpha': 65})
#    print f_max_target.subs({'theta': 70, 'alpha': 65})
#    print f_max_target.subs({'theta': 65, 'alpha': 65})
#    print f_max_target.subs({'theta': 60, 'alpha': 65})
#    print f_max_target.subs({'theta': 55, 'alpha': 65})
#    print f_max_target.subs({'theta': 50, 'alpha': 65})
#    print '60!'
#    print f_max_target.subs({'theta': 80, 'alpha': 60})
#    print f_max_target.subs({'theta': 75, 'alpha': 60})
#    print f_max_target.subs({'theta': 70, 'alpha': 60})
#    print f_max_target.subs({'theta': 65, 'alpha': 60})
#    print f_max_target.subs({'theta': 60, 'alpha': 60})
#    print f_max_target.subs({'theta': 55, 'alpha': 60})
#    print f_max_target.subs({'theta': 50, 'alpha': 60})
#    print '!!!!!'
#    print f_max_target.subs({'theta': 40, 'alpha': 85})
#    print f_max_target.subs({'theta': 40, 'alpha': 80})
#    print f_max_target.subs({'theta': 40, 'alpha': 75})
#    print f_max_target.subs({'theta': 40, 'alpha': 60})
#    print f_max_target.subs({'theta': 40, 'alpha': 55})
#    print f_max_target.subs({'theta': 40, 'alpha': 50})
#    print f_max_target.subs({'theta': 40, 'alpha': 45})
#    print '90!'
#    print f_max_target.subs({'theta': 30, 'alpha': 90})
#    print f_max_target.subs({'theta': 25, 'alpha': 90})
#    print f_max_target.subs({'theta': 20, 'alpha': 90})
#    print f_max_target.subs({'theta': 15, 'alpha': 90})
#    print f_max_target.subs({'theta': 10, 'alpha': 90})
#    print f_max_target.subs({'theta': 5, 'alpha': 90})
#    print f_max_target.subs({'theta': 2, 'alpha': 90})
#    print '85!'
#    print f_max_target.subs({'theta': 30, 'alpha': 85})
#    print f_max_target.subs({'theta': 25, 'alpha': 85})
#    print f_max_target.subs({'theta': 20, 'alpha': 85})
#    print f_max_target.subs({'theta': 15, 'alpha': 85})
#    print f_max_target.subs({'theta': 10, 'alpha': 85})
#    print f_max_target.subs({'theta': 5, 'alpha': 85})
#    print f_max_target.subs({'theta': 2, 'alpha': 85})
#    print '80!'
#    print f_max_target.subs({'theta': 30, 'alpha': 80})
#    print f_max_target.subs({'theta': 25, 'alpha': 80})
#    print f_max_target.subs({'theta': 20, 'alpha': 80})
#    print f_max_target.subs({'theta': 15, 'alpha': 80})
#    print f_max_target.subs({'theta': 10, 'alpha': 80})
#    print f_max_target.subs({'theta': 5, 'alpha': 80})
#    print f_max_target.subs({'theta': 2, 'alpha': 80})
#    print '75!'
#    print f_max_target.subs({'theta': 30, 'alpha': 75})
#    print f_max_target.subs({'theta': 25, 'alpha': 75})
#    print f_max_target.subs({'theta': 20, 'alpha': 75})
#    print f_max_target.subs({'theta': 15, 'alpha': 75})
#    print f_max_target.subs({'theta': 10, 'alpha': 75})
#    print f_max_target.subs({'theta': 5, 'alpha': 75})
#    print f_max_target.subs({'theta': 2, 'alpha': 75})
#    print '70!'
#    print f_max_target.subs({'theta': 30, 'alpha': 70})
#    print f_max_target.subs({'theta': 25, 'alpha': 70})
#    print f_max_target.subs({'theta': 20, 'alpha': 70})
#    print f_max_target.subs({'theta': 15, 'alpha': 70})
#    print f_max_target.subs({'theta': 10, 'alpha': 70})
#    print f_max_target.subs({'theta': 5, 'alpha': 70})
#    print f_max_target.subs({'theta': 2, 'alpha': 70})
#    print '65!'
#    print f_max_target.subs({'theta': 30, 'alpha': 65})
#    print f_max_target.subs({'theta': 25, 'alpha': 65})
#    print f_max_target.subs({'theta': 20, 'alpha': 65})
#    print f_max_target.subs({'theta': 15, 'alpha': 65})
#    print f_max_target.subs({'theta': 10, 'alpha': 65})
#    print f_max_target.subs({'theta': 5, 'alpha': 65})
#    print f_max_target.subs({'theta': 2, 'alpha': 65})
    return f_max_target         

def make_posterior_function(all_table_partion_list, all_cut_proportion_list):
    table_partion_num = len(all_table_partion_list)                
    theta = Symbol('theta')
    alpha = Symbol('alpha')
    f_prior = Function('f_prior')
    f_likelihood = Function('f_likelihood')
    f_posterior = Function('f_posterior')
    f_prior = 1
    f_likelihood = 1
    d = 1
    for i in range(table_partion_num):
        guest_total_num = np.array(all_table_partion_list[i]).sum()
        table_num = len(all_table_partion_list[i])
        permutation_table = list(set(list(itertools.permutations(all_table_partion_list[i]))))
        
        comb_total_num = 0
        for j in range(len(permutation_table)):
            guest_arrange_num = guest_total_num
            comb_num = 1
            for jj in range(len(permutation_table[j])):    
                guest_num_per_table = permutation_table[j][jj]
                comb_num = comb_num * int(comb(guest_arrange_num - 1, guest_num_per_table - 1))
                guest_arrange_num = guest_arrange_num - guest_num_per_table
            comb_total_num = comb_total_num + comb_num
            
        f_prior = f_prior * gamma(theta) / gamma(theta + guest_total_num) \
                  * comb_total_num \
                  / (1 - gamma(theta) / gamma(theta + guest_total_num) * theta * gamma(guest_total_num)) \
                  / len(permutation_table) 
                  
        f_likelihood = f_likelihood * gamma(table_num * alpha) / (gamma(alpha) ** table_num) * 0.5
        
        for k in range(table_num):
            guest_num = all_table_partion_list[i][k]
            cut_proportion = all_cut_proportion_list[i][k]
            
            f_prior = f_prior * gamma(guest_num) * theta
            f_likelihood = f_likelihood * (cut_proportion ** (alpha - 1))
    
    f_posterior = f_prior * f_likelihood 
#    print f_prior.subs({'theta': 1.0, 'alpha': 10.0})
#    print f_posterior.subs({'theta': 2.0, 'alpha': 20.0}) / f_posterior.subs({'theta': 0.967, 'alpha': 21.0})
#    print f_posterior.subs({'theta': 2.0, 'alpha': 20.0}) 
#    print f_posterior.subs({'theta': 0.967, 'alpha': 21.0})
#    print 'ttt'
#    print '!!!', f_prior, f_likelihood
    return f_posterior

def get_derivative(function, argument):
    print '^^^'
    return diff(function, argument)  

def gradient_ascent(f_target, argument_symbols, argument_value_ranges, tolerance, iter_num, max_steps_num): 
    step_sizes = [100, 10, 1, 0.1, 0.01, 0.001]
    curr_f_value = float('-inf')
    
    f_gradient = map(get_derivative, [f_target] * len(argument_symbols), argument_symbols)
    print '111'
        
    for i in range(iter_num):
        print i
        argument_value_dict = {}
        for j in range(len(argument_symbols)):
            argument_value_range = argument_value_ranges[j]
            argument_value_sample = random.uniform(argument_value_range[0], argument_value_range[1])
            argument_value_dict[argument_symbols[j]] = argument_value_sample
        f_value = f_target.subs(argument_value_dict)
        print f_value
        steps_num = 0
        
        argument_value_dict_list = []
        while True:
#            print f_gradient
            gradient = []
            for k in range(len(argument_symbols)):
                gradient_partial = f_gradient[k].subs(argument_value_dict)
                gradient_partial = round(gradient_partial, 10)
#                print gradient_partial
#                print gradient_partial
                next_argument_values = map(round, map(lambda x: argument_value_dict[argument_symbols[k]] + gradient_partial * x, step_sizes), [10] * len(step_sizes))
#                print next_argument_values
                ###positive ensurence
                nonpositive_argument_index = [next_argument_values.index(l) for l in next_argument_values if l <= 0]
                next_argument_values = np.array(next_argument_values)
                next_argument_values[nonpositive_argument_index] = argument_value_ranges[k][0]
                next_argument_values = list(next_argument_values)
                
                for m in range(len(next_argument_values)):
                    if k == 0:
                        argument_value_dict_list.append({})
                    
                    argument_value_dict_list[m][argument_symbols[k]] = next_argument_values[m]
#            print '!!!',  argument_value_dict_list  
            next_f_values = map(round, map(lambda x: f_target.subs(x), argument_value_dict_list), [10] * len(step_sizes))
#            print '&&&', next_argument_values, next_f_values, next_f_values.index(max(next_f_values))
            next_f_value = max(next_f_values)
            next_argument_value = argument_value_dict_list[next_f_values.index(max(next_f_values))]
#            print '***', argument_value_dict_list, next_f_values
            print '!!!', next_argument_value, next_f_value
#            if (next_argument_value < argument_value_range[0]):
#                argument_value = argument_value_range[0]
#                f_value = f_target.subs(argument_symbol, argument_value)
#                break
#            
#            if (next_argument_value > argument_value_range[1]):
#                argument_value = argument_value_range[1]
#                f_value = f_target.subs(argument_symbol, argument_value)
#                break

            if (((next_f_value - f_value)/abs(f_value)) * 1.0 < tolerance) or (steps_num > max_steps_num):
                break  
            else:
                argument_value_dict, f_value = next_argument_value, next_f_value
                argument_value_dict_list = []
                steps_num = steps_num + 1
#                print '**', steps_num
#                print argument_value
           
            if steps_num%10 == 0:
                print steps_num
        if f_value > curr_f_value:
            curr_argument_value = argument_value_dict
            curr_f_value = f_value
            print curr_argument_value, i, steps_num, curr_f_value
            
    return curr_argument_value, curr_f_value
            
        
def main():           
    f_max_target = make_target_function(graph_num = GRAPH_NUM, theta_range_num = THETA_RANGE_NUM, alpha_range_num = ALPHA_RANGE_NUM)
    theta = Symbol('theta')
    alpha = Symbol('alpha')
#    print f_max_target
#    print '*********************'
    
#    derivative_prior = get_derivative(f_prior, theta)
#    derivative_likelihood = get_derivative(f_likelihood, alpha)
    
    best_paramete, best_value = gradient_ascent(f_target = f_max_target, argument_symbols = [theta, alpha], argument_value_ranges = [[MIN_THETA, MAX_THETA], [MIN_ALPHA, MAX_ALPHA]], tolerance = TOLERANCE, iter_num = ITER_NUM, max_steps_num = MAX_STEPS_NUM)    
    
#    print 'ooo', get_derivative(f_prior, theta), f_likelihood
    
if __name__ == "__main__":
    main()