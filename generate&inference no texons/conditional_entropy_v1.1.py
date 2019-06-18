# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 13:52:00 2017

output: no all_possible_infer_trees, only most_likely_tree

@author: Edward Coen
"""

from __future__ import division
from mpl_toolkits.mplot3d import Axes3D
import scipy.special
from scipy.misc import comb
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
import cPickle as pickle
from sympy import *
import partial_corr as pc

sys.setrecursionlimit(1000000000)
"""
number of img to infer
"""
#SUBJECT_NUM_LIST = range(1, 26) + range(101, 104) + range(105, 111) + range(201, 209) + range(301, 311)
IMG_LIST = [8, 20, 31, 49, 85, 99, 112, 118, 142, 153, 184, 223, 380, 411, 444, 575, 623, 639, 817, 834]
Z_SCORE = np.array([0.373, 2.472, -6.604, -22.068, -0.836, 0.232, -5.541, 4.786, 1.073, 5.833, -11.927, 12.288, 7.737, 7.652, -12.932, 9.322, 1.878, -9.983, 2.900, 8.438])
SIZE_STD = np.array([7.765, 2.084, 8.892, 2.354, 7.642, 10.444, 4.270, 3.293, 11.617, 8.348, 4.946, 12.209, 9.588, 10.769, 6.952, 13.229, 9.573, 10.573, 13.298, 14.732])
IMG_NUM_BATCH = 2
IMG_NUM = 10001

"""
ncrp parameters
"""

GAMMA = 0.9
ALL_GUEST_NUM = 6

"""
image parameters
"""

IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 768
COLOR_SPACE = [[128,128,128],[0,0,255],[0,255,0],[255,0,0],[0,255,255],[255,0,255],[255,255,0],[255,255,255]]
GRAPH_WIDTH = 500
TREE_GRAPH_HEIGHT = 252
INFER_IMG_ADJUST_HEIGHT = 300

"""
Dirchlet parmeters
"""
ALPHA_BASE = [3.5]


"""
code parameters
"""

"""
global arguements
"""

#GRID_SIZE = 10

#CODE_BASE = 10    #decimal
#
#class decoder():    
#    
#    def __init__(self, code_base):
#        self.code_base = code_base
#        
#    def decode_princeple(self, all_guest_num): 
#        curr_table_guest_num = self.code_tree[0]
#        self.code_tree = self.code_tree[1:]
#            
#        self.curr_guest_num = self.curr_guest_num + int(curr_table_guest_num)
#        self.table_partion_list[len(self.table_partion_list) - 1].append(int(curr_table_guest_num))
#        
#        if int(curr_table_guest_num) != 1:
#            self.all_guest_num_list.append(int(curr_table_guest_num))
#
#        if self.curr_guest_num == all_guest_num:
#            self.table_partion_list.append([])
#            self.curr_guest_num = 0 
#            return
#        else:
#            return self.decode_princeple(all_guest_num)
#
#    def make_decode_list(self, code_tree):
#        self.code_tree = code_tree
#        self.table_partion_list = [[]]
#        self.all_guest_num_list = [ALL_GUEST_NUM]
#        self.curr_guest_num = 0
#        map(self.decode_princeple, self.all_guest_num_list)
#        del self.table_partion_list[-1]
#        self.all_table_partion_list.append(self.table_partion_list)
#        
#    def __call__(self):
#        self.code_tree_list = list(pd.read_csv('E:/ncrp_generate/tree_kind_num_' + str(ALL_GUEST_NUM) + '.csv')['tree_code'])
#        self.code_tree_list = map(str, self.code_tree_list)        
#        self.all_table_partion_list = []  
#        map(self.make_decode_list, self.code_tree_list)
#        return self.all_table_partion_list
    
""" possible tree and its p value for curr img """
    
class possible_tree_generator():  
    def __init__(self, img_num):
        self.img_num = img_num
        self.all_img_tree_list = []
        
    def __call__(self):
        img_original = cv2.imread('E:/ncrp_generate/' + str(10001 + IMG_LIST[self.img_num]) + '.png')
        img_init = img_original
#        cv2.namedWindow('image')
#        cv2.imshow('image', img_init)
#        cv2.waitKey()
#        cv2.destroyAllWindows()
        tree_init = nx.DiGraph()
        tree_init.add_node(1, img = img_init, complete = 0, terminal = 0, p_like = 1, p_poster_list = [], guest = [], depth = 1, possible_trees_num = 0)
        tree_list_init = [tree_init]
        possible_tree_list = possible_tree(tree_list_init)
        ncrp_tree_list = map(ncrp_tree_transform, possible_tree_list)
        return ncrp_tree_list
        
def possible_tree(tree_list): 
    all_tree_complete = map(lambda x: x.node[1]['complete'], tree_list)
    if all_tree_complete.count(0) == 0:
        return tree_list
    
    else:
        tree_list_temp = map(tree_extend, tree_list)
        tree_list = tree_flatten(tree_list_temp)
        return possible_tree(tree_list)

def tree_extend(tree):
    if tree.node[1]['complete'] == 1:
        return [tree]
    
    else:
        leaf_nodes = [ n for n,d in tree.out_degree().items() if d==0]
        leaf_nodes_terminal = map(lambda x: tree.node[x]['terminal'], leaf_nodes)
        if leaf_nodes_terminal.count(0) == 0:
            tree.node[1]['complete'] = 1
            return [tree]
        
        else:
            extend_trees = node_extend(tree, len(tree.nodes()), leaf_nodes, 0)
            return extend_trees
    
def node_extend(tree, new_node_id_start, leaf_nodes, curr_leaf_node_index):
    if tree.node[leaf_nodes[curr_leaf_node_index]]['terminal'] == 1:
        return node_extend(tree, new_node_id_start, leaf_nodes, curr_leaf_node_index + 1)
    else:
        img_to_cut = tree.node[leaf_nodes[curr_leaf_node_index]]['img'] 
        possible_cut_edges = cal_possible_cut_edges(img_to_cut)
        extended_trees = make_extended_tree(tree, possible_cut_edges, new_node_id_start, leaf_nodes[curr_leaf_node_index])
        return extended_trees

def make_extended_tree(tree, possible_cut_edges, new_node_id_start, curr_leaf_node):
    edges_vertical = possible_cut_edges[0]
    edges_horizontal = possible_cut_edges[1]
    img_to_cut = tree.node[curr_leaf_node]['img']
    img_height, img_width, channel_num = img_to_cut.shape    
    if len(edges_vertical) == 2 and len(edges_horizontal) == 2:
        tree.node[curr_leaf_node]['terminal'] = 1
        return [tree]        
        
    else:
        extended_trees = []
    
        if len(edges_vertical) != 2:
            for i in range(len(edges_vertical) - 2):
                combination = list(itertools.combinations(range(len(edges_vertical) - 2), i + 1))
                combination_list = map(list, combination)

                for z in range(len(combination_list)):
                    tree_temp = tree.copy()
                    cut_propotion = []
                    
                    tree_temp.add_node(new_node_id_start + 1, img = img_to_cut[0:img_height, edges_vertical[0] + 1:edges_vertical[combination_list[z][0] + 1]], terminal = 0, cut_proportion = [], guest = [], depth = tree.node[curr_leaf_node]['depth'] + 1)
                    tree_temp.add_edge(curr_leaf_node , new_node_id_start + 1)
                    cut_propotion.append((edges_vertical[combination_list[z][0] + 1] - (edges_vertical[0] + 1) + 1)/img_width)
                    if len(combination_list[z]) != 1:
                        for j in range(len(combination_list[z]) - 1):
                            tree_temp.add_node(new_node_id_start + 1 + (j + 1), img = img_to_cut[0:img_height, (edges_vertical[combination_list[z][j] + 1] + 1):edges_vertical[combination_list[z][j + 1] + 1]], terminal = 0, cut_proportion = [], guest = [], depth = tree.node[curr_leaf_node]['depth'] + 1)
                            tree_temp.add_edge(curr_leaf_node, new_node_id_start + 1 + (j + 1))
                            cut_propotion.append((edges_vertical[combination_list[z][j + 1] + 1] - (edges_vertical[combination_list[z][j] + 1] + 1) + 1)/img_width) 
                            
                    tree_temp.add_node(new_node_id_start + len(combination_list[z]) + 1, img = img_to_cut[0:img_height, edges_vertical[combination_list[z][-1] + 1] + 1:edges_vertical[-1]], terminal = 0, cut_proportion = [], guest = [], depth = tree.node[curr_leaf_node]['depth'] + 1)
                    tree_temp.add_edge(curr_leaf_node , new_node_id_start + len(combination_list[z]) + 1)
                    cut_propotion.append(1 - np.array(cut_propotion).sum())
                    
                    tree_temp.node[curr_leaf_node]['cut_proportion'] = cut_propotion
#                    tree_temp.node[1]['p_like'] = tree_temp.node[1]['p_like']*cal_p_dirichlet(cut_propotion)*0.5
                    extended_trees.append(tree_temp)
                    
        else:
            for i in range(len(edges_horizontal) - 2):
                combination = list(itertools.combinations(range(len(edges_horizontal) - 2), i + 1))
                combination_list = map(list, combination)
                for z in range(len(combination_list)):
                    tree_temp = tree.copy()
                    cut_propotion = []
                    
                    tree_temp.add_node(new_node_id_start + 1, img = img_to_cut[edges_vertical[0] + 1:edges_horizontal[combination_list[z][0] + 1], 0:img_width], terminal = 0, cut_proportion = [], guest = [], depth = tree.node[curr_leaf_node]['depth'] + 1)
                    tree_temp.add_edge(curr_leaf_node , new_node_id_start + 1)
                    cut_propotion.append((edges_horizontal[combination_list[z][0] + 1] - (edges_horizontal[0] + 1) + 1)/img_height)
                    if len(combination_list[z]) != 1:
                        for j in range(len(combination_list[z]) - 1):
                            tree_temp.add_node(new_node_id_start + 1 + (j + 1), img = img_to_cut[(edges_horizontal[combination_list[z][j] + 1] + 1):edges_horizontal[combination_list[z][j + 1] + 1], 0:img_width], terminal = 0, cut_proportion = [], guest = [], depth = tree.node[curr_leaf_node]['depth'] + 1)
                            tree_temp.add_edge(curr_leaf_node, new_node_id_start + + 1 + (j + 1))
                            cut_propotion.append((edges_horizontal[combination_list[z][j + 1] + 1] - (edges_horizontal[combination_list[z][j] + 1] + 1) + 1)/img_height)       
                    
                    tree_temp.add_node(new_node_id_start + len(combination_list[z]) + 1, img = img_to_cut[edges_horizontal[combination_list[z][-1] + 1] + 1:edges_horizontal[-1], 0:img_width], terminal = 0, cut_proportion = [], guest = [], depth = tree.node[curr_leaf_node]['depth'] + 1)
                    tree_temp.add_edge(curr_leaf_node , new_node_id_start + len(combination_list[z]) + 1)
                    cut_propotion.append(1 - np.array(cut_propotion).sum())
                    
                    tree_temp.node[curr_leaf_node]['cut_proportion'] = cut_propotion
#                    tree_temp.node[1]['p_like'] = tree_temp.node[1]['p_like']*cal_p_dirichlet(cut_propotion)*0.5
                    extended_trees.append(tree_temp)              
        return extended_trees

def cal_possible_cut_edges(img): 
    img_height, img_width, channel_num = img.shape
    edges_vertical = [-1]
    edges_horizontal = [-1]
    for i in range(4, img_width - 3):
        if (img[0][i][0] != img[0][i + 1][0] or img[0][i][1] != img[0][i + 1][1] or img[0][i][2] != img[0][i + 1][2]):  
            for k in range(i-2, i+3):
                if (img[img_height - 1][k][0] != img[img_height - 1][k + 1][0] or img[img_height - 1][k][1] != img[img_height - 1][k + 1][1] or img[img_height - 1][k][2] != img[img_height - 1][k + 1][2]):
                    edges_vertical.append(i)
                    break
    edges_vertical.append(img_width - 1)
    
    for j in range(4, img_height - 3):
        if (img[j][0][0] != img[j + 1][0][0] or img[j][0][1] != img[j + 1][0][1] or img[j][0][2] != img[j + 1][0][2]):
            for l in range(j-2, j+3):
                if (img[l][img_width - 1][0] != img[l + 1][img_width - 1][0] or img[l][img_width - 1][1] != img[l + 1][img_width - 1][1] or img[l][img_width - 1][2] != img[l + 1][img_width - 1][2]):   
                    edges_horizontal.append(j)
                    break
    edges_horizontal.append(img_height - 1)

    return edges_vertical, edges_horizontal
 
#def cal_possible_cut_edges(img): 
#    img_height, img_width, channel_num = img.shape
#    edges_vertical = [-1]
#    edges_horizontal = [-1]
#    for i in range(5, img_width - 4):
#        if (img[5][i][2] == 0 and img[5][i + 1][2] == 255):  
#            for k in range(i-2, i+3):
#                if (img[img_height - 5][k][2] == 0 and img[img_height - 5][k + 1][2] == 255):
#                    edges_vertical.append(i)
#                    break
#    edges_vertical.append(img_width - 1)
#    
#    for j in range(5, img_height - 4):
#        if (img[j][5][2] == 0 and img[j + 1][5][2] == 255):
#            for l in range(j-2, j+3):
#                if (img[l][img_width - 5][2] == 0 and img[l + 1][img_width - 5][2] == 255):   
#                    edges_horizontal.append(j)
#                    break
#    edges_horizontal.append(img_height - 1)
#
#    return edges_vertical, edges_horizontal

def cal_p_dirichlet(cut_propotion):
    alpha = ALPHA_BASE * len(cut_propotion)
    return scipy.stats.dirichlet.pdf(cut_propotion, alpha)

def tree_flatten(tree_list):
    flattened_tree_list = []
    for i in range(len(tree_list)):
        for k in range(len(tree_list[i])):
            flattened_tree_list.append(tree_list[i][k])
    return flattened_tree_list

def ncrp_tree_transform(tree):
    nonleaf_nodes = [ n for n,d in tree.out_degree().items() if d!=0]
    for i in range(len(tree.nodes()), 0, -1):
        if i not in nonleaf_nodes:
            tree.node[i]['guest'] = [i]
        else:
            guest_list = []
            children = nx.DiGraph.successors(tree, i)
            map(lambda x: guest_list.extend(tree.node[x]['guest']), children)
            tree.node[i]['guest'] = guest_list
    return tree

"""transform possible tree to table partion list  """
"""make the prior probability calculation easy"""
"""make the likelihood probabilty list"""

class likelihood():
    def __init__(self, all_tree_graph):

        self.all_tree_code = []
        self.all_tree_p = []
        self.all_tree_graph = all_tree_graph
        
    def prior_code_tree_transform(self, tree_graph):
        self.graph = tree_graph
        self.code_list = [[ALL_GUEST_NUM]]
        self.to_process_node_id = [1]
        map(self.cal_code_list, self.to_process_node_id)
        self.all_tree_code.append(self.code_list)
        self.all_tree_p.append(tree_graph.node[1]['p_like'])
            
    def cal_code_list(self, node_id):
            
        children = nx.DiGraph.successors(self.graph, node_id)
        if children:
            self.element_sorted_for = map(lambda x: (len(self.graph.node[x]['guest']), self.cal_children_single_guest_num(x)), children)
            self.element_no_repeat = list(set(self.element_sorted_for))
            self.element_no_repeat.sort()
            self.element_with_index_list = list(enumerate(self.element_sorted_for))
            self.element_sorted_for.sort()
            element_to_append = list(np.array(self.element_sorted_for)[:, 0])
            children_index_sorted = np.array(map(self.cal_index_sorted, self.element_no_repeat))
            children_index = filter(lambda x: x != None, children_index_sorted.flatten())
            children_sorted = map(lambda x: children[x], children_index)
            
            self.to_process_node_id.extend(children_sorted)
            self.code_list.append(element_to_append)

    def cal_children_single_guest_num(self, node_id):
        single_guest_num_list =  map(lambda x: len(self.graph.node[x]['guest']), nx.DiGraph.successors(self.graph, node_id))
        return len(single_guest_num_list) - single_guest_num_list.count(1) 
    
    def cal_index_sorted(self, element):
        self.target = element
        return map(self.find_index, range(len(self.element_with_index_list))) 
        
    def find_index(self, index):
        if self.element_with_index_list[index][1] == self.target:
            return index

    def __call__(self):
        map(self.prior_code_tree_transform, self.all_tree_graph)
        return self.all_tree_code, self.all_tree_p

"""cal the prior probability for every possible tree of curr img by the 'table partion list' form"""
    
class prior():  
    
    def __init__(self, all_table_partion_list):
        self.all_table_partion_list = all_table_partion_list
    
    def cal_renomalize_parameter(self, table_partion):
        return 1/((1 - scipy.special.gamma(GAMMA) * GAMMA * scipy.special.gamma(np.array(table_partion).sum()) / scipy.special.gamma(np.array(table_partion).sum() + GAMMA)) * len(list(set(itertools.permutations(np.array(table_partion)))))) 
        
    def cal_probability_table_partion(self, table_partion):
#        print table_partion
#        print reduce(op.mul, map(scipy.special.gamma, np.array(table_partion))) * scipy.special.gamma(GAMMA) * pow(GAMMA, len(table_partion)) / scipy.special.gamma(np.array(table_partion).sum() + GAMMA)
        return reduce(op.mul, map(scipy.special.gamma, np.array(table_partion))) * scipy.special.gamma(GAMMA) * pow(GAMMA, len(table_partion)) / scipy.special.gamma(np.array(table_partion).sum() + GAMMA)
            
    def cal_permutation_table_partion(self, table_partion):
        return list(set(list(itertools.permutations(table_partion))))
    
    def cal_all_combination_guest(self, permutation_table_partion): 
        return reduce(op.add, map(self.cal_permutation_combination_guest, permutation_table_partion))
        
    def cal_permutation_combination_guest(self, table_partion):
        self.guest_left = np.array(table_partion).sum()
        return reduce(op.mul, map(self.cal_combination_guest, table_partion))
        
    def cal_combination_guest(self, table_guest_num):
        combination_num = round(comb(self.guest_left - 1, table_guest_num - 1))
        self.guest_left = self.guest_left - table_guest_num
        return combination_num

    def cal_prior_probability(self, table_partion_list):
        probability_table_partion = map(self.cal_probability_table_partion, table_partion_list[1:])
        permutation_table_partion = map(self.cal_permutation_table_partion, table_partion_list[1:])
        all_combination_guest = map(self.cal_all_combination_guest, permutation_table_partion)
        renomalize_parameter = map(self.cal_renomalize_parameter, table_partion_list[1:])
#        print 'ttt', all_combination_guest, renomalize_parameter
#        print reduce(op.mul, np.array(probability_table_partion)*np.array(all_combination_guest)*np.array(renomalize_parameter))
        return reduce(op.mul, np.array(probability_table_partion)*np.array(all_combination_guest)*np.array(renomalize_parameter))
        
    def __call__(self):
        return map(self.cal_prior_probability, self.all_table_partion_list)
    
""" make img likelihood function in abstract form"""

def make_posterior_function(possible_tree):
    g = possible_tree
    nonleaf_nodes = [ n for n,d in g.out_degree().items() if d!=0]                 
    all_table_partion_list = map(lambda x: map(lambda y: len(g.node[y]['guest']), nx.DiGraph.successors(g, x)), nonleaf_nodes)
    all_cut_proportion_list = map(lambda x: g.node[x]['cut_proportion'], nonleaf_nodes)
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
    return f_posterior
    
"""cal the posterior probability and visualization the most likely tree"""
class posterior():
    def __init__(self, img_num):
        self.img_num = img_num
        
    def __call__(self):
#        img_loglikelihood_given_gamma_alpha = 1
        f_img_likelihood = Function('f_img_likelihood')
        f_all_img = []
        for z in range(self.img_num):
            print z
            Possible_Tree = possible_tree_generator(z)
            
            Possible_Tree_List = Possible_Tree()
            f_img_likelihood = reduce(op.add, map(make_posterior_function, Possible_Tree_List))
            f_all_img.append(f_img_likelihood)
#            visualization(Possible_Tree_List[0], z)
#            Likelihood = likelihood(all_tree_graph = Possible_Tree_List)
#            Likelihood_Probability = Likelihood()
#            print len(Likelihood_Probability[0])          
#            All_Table_Partion_List = map(lambda x: Likelihood_Probability[0][x], range(len(Likelihood_Probability[0])))
#            Prior = prior(all_table_partion_list = All_Table_Partion_List)
#            Prior_Probability =  Prior()
#            posterior_list = map(lambda x: Prior_Probability[x] * Likelihood_Probability[1][x], range(len(Likelihood_Probability[0])))
#            print np.array(posterior_list).sum()
#            print np.log2(np.array(posterior_list).sum())
#            img_loglikelihood_given_gamma_alpha = img_loglikelihood_given_gamma_alpha*(np.sum(posterior_list))
        return f_all_img
#            posterior_sample_list = map(lambda x: x * 1.0 / np.array(posterior_list).sum(), posterior_list)
##            max_index = posterior_list.index(np.array(posterior_list).max())
##            most_like_tree = Possible_Tree_List[max_index]
##            most_like_tree.node[1]['possible_trees_num'] = len(Possible_Tree_List)
##            most_like_tree.node[1]['p_poster_list'] = posterior_list
            
##            num_graph_col = int((len(Possible_Tree_List) - 1)/3) + 1
#            tree_graph_width = int(TREE_GRAPH_HEIGHT * 1.5 * 3 / num_graph_col)
#    
#            img_trees = np.ones([TREE_GRAPH_HEIGHT * 3, int(TREE_GRAPH_HEIGHT * 1.5 * 3) + 481, 3], 'uint8') * 255
            
##            for i in range(3):
##                for j in range(num_graph_col):
##                    if i * num_graph_col + j < len(Possible_Tree_List):
#                        position = nx.drawing.nx_agraph.graphviz_layout(Possible_Tree_List[i * num_graph_col + j], prog = 'dot')
#                        color_selections = [(190/256,0,0)] * len(Possible_Tree_List[i * num_graph_col + j].nodes())
#                        label_selections = [str(len(Possible_Tree_List[i * num_graph_col + j].node[n]['guest'])) for n in Possible_Tree_List[i * num_graph_col + j].nodes()]
#                        leaf_nodes = [ n for n,d in Possible_Tree_List[i * num_graph_col + j].out_degree().items() if d==0]
#                        for t in range(len(leaf_nodes)):
#                            b,g,r = Possible_Tree_List[i * num_graph_col + j].node[leaf_nodes[t]]['img'][0][0]
#                            color_selections[leaf_nodes[t] - 1] = (r/256, g/256, b/256)
#                            
#                        nx.draw_networkx_nodes(Possible_Tree_List[i*num_graph_col + j], pos = position, node_color = color_selections, label = label_selections)
#                        nx.draw_networkx_edges(Possible_Tree_List[i*num_graph_col + j], pos = position)
#                        plt.savefig('E:/ncrp_test/'+str(i * num_graph_col + j)+'.png')
#                        plt.close()
##                        nx.write_gpickle(Possible_Tree_List[i * num_graph_col + j], 'E:/ncrp_infer/possible_tree_infer_alpha'+str(ALPHA_BASE)+'_gamma['+str(GAMMA)+']_img_'+str(z)+'_tree_'+str(i*num_graph_col+j)+'.gpickle')
                        
#                        tree_graph_img = cv2.imread('E:/ncrp_test/'+str(i * num_graph_col + j)+'.png')
#                        adjust_tree_graph_img = cv2.resize(tree_graph_img, (tree_graph_width, TREE_GRAPH_HEIGHT), interpolation = cv2.INTER_CUBIC)
#                        img_trees[TREE_GRAPH_HEIGHT * i:TREE_GRAPH_HEIGHT * (i + 1), tree_graph_width * j:tree_graph_width * (j + 1)] = adjust_tree_graph_img
#    
#    
#            img_to_infer = cv2.imread('E:/ncrp/'+str(z)+'.png')
#            adjust_img_to_infer = cv2.resize(img_to_infer, (int(INFER_IMG_ADJUST_HEIGHT * 4/3), INFER_IMG_ADJUST_HEIGHT), interpolation = cv2.INTER_CUBIC)
#            img_trees[0:INFER_IMG_ADJUST_HEIGHT, int(TREE_GRAPH_HEIGHT * 1.5 * 3) + 1:int(TREE_GRAPH_HEIGHT * 1.5 * 3 + INFER_IMG_ADJUST_HEIGHT * 4/3) + 1] = adjust_img_to_infer
#            
#            x = np.arange(len(posterior_list))          
#            y = posterior_list          
#            plt.bar(x, y, alpha = .9, color = 'g')
#            plt.savefig('E:/ncrp_test/hist.png')   
#            plt.close()
#            img_hist = cv2.imread('E:/ncrp_test/hist.png')
#            adjust_img_hist = cv2.resize(img_hist, (int(TREE_GRAPH_HEIGHT * 1.5), TREE_GRAPH_HEIGHT), interpolation = cv2.INTER_CUBIC)
#            img_trees[TREE_GRAPH_HEIGHT * 2:TREE_GRAPH_HEIGHT * 3, int(TREE_GRAPH_HEIGHT * 1.5 * 3) + 1:int(TREE_GRAPH_HEIGHT * 1.5 * 4) + 1] = adjust_img_hist 
#            
#            row_index, col_index = divmod(max_index, num_graph_col)
#            cv2.rectangle(img_trees, (int(col_index * tree_graph_width), int(row_index * TREE_GRAPH_HEIGHT)), (int((col_index + 1) * tree_graph_width), int((row_index + 1) * TREE_GRAPH_HEIGHT)), color = (0, 0, 255)) 
#            
#            cv2.namedWindow('inference_demo')
#            cv2.imshow('inference_demo', img_trees)
#            cv2.waitKey()
#            cv2.imwrite('E:/ncrp_infer/trees_hist_alpha'+str(ALPHA_BASE)+'_gamma['+str(GAMMA)+']_'+str(z)+'.png', img_trees)
#            cv2.destroyAllWindows()
#            
#            
#            print 'ok!!', most_like_tree.nodes(), most_like_tree.edges()
##            nx.write_gpickle(most_like_tree, 'E:/ncrp_infer/most_like_tree_alpha'+str(ALPHA_BASE)+'_gamma['+str(GAMMA)+']_'+str(z)+'.gpickle')
#            visualization(most_like_tree, z)
    
def visualization(most_like_tree, img_num):
    cv2.namedWindow('inference_demo')
    img_show = np.zeros([IMAGE_HEIGHT, IMAGE_WIDTH + GRAPH_WIDTH, 3], 'uint8')
    img_to_infer = cv2.imread('E:/redraw/blank1.0/'+IMG_LIST[img_num]+'.png')
    img_show[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH] = img_to_infer[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]
    
    intervel_x_list = [0, GRAPH_WIDTH]
    intervel_y = IMAGE_HEIGHT/(ALL_GUEST_NUM + 1)
    node_draw_x_list = [0, GRAPH_WIDTH/2]
    
    node_radius = 15
    font_parm = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
    
    for i in range(1, len(most_like_tree.nodes()) + 1):
        depth = most_like_tree.node[i]['depth'] 
        
        children = nx.DiGraph.successors(most_like_tree, i)
        
        if children:
            intervel_x = intervel_x_list[i]
            intervel_x_list.extend([intervel_x/len(children)] * len(children))
            center_node_x = node_draw_x_list[i]
            left_intervel_x = center_node_x - intervel_x/2
            node_draw_x = map(lambda x: left_intervel_x + (children.index(x) + 1)/(len(children) + 1) * intervel_x, children)
            node_draw_x_list.extend(node_draw_x)
            
            cv2.circle(img_show, (int(node_draw_x_list[i] + IMAGE_WIDTH), int(intervel_y * depth)), node_radius, (255, 255, 255), -1)
            cv2.putText(img_show, 
                        str(len(most_like_tree.node[i]['guest'])), 
                           (int(node_draw_x_list[i] + IMAGE_WIDTH - node_radius/2) , int(intervel_y * depth + node_radius/2)), 
                           font_parm, 0.8, (0,  0, 190), 1)
            for j in range(len(children)):
                cv2.line(img_show, 
                         (int(node_draw_x_list[i] + IMAGE_WIDTH), int(intervel_y * depth + node_radius)),
                         (int(node_draw_x_list[children[j]] + IMAGE_WIDTH), int(intervel_y * (depth + 1) - node_radius)),
                         (255, 255, 255), 2)
        else:
            b,g,r = most_like_tree.node[i]['img'][0][0]
            cv2.circle(img_show, (int(node_draw_x_list[i] + IMAGE_WIDTH), int(intervel_y * depth)), node_radius, (int(b), int(g), int(r)), -1)
            cv2.putText(img_show, 
                        str(len(most_like_tree.node[i]['guest'])), 
                        (int(node_draw_x_list[i] + IMAGE_WIDTH - node_radius/2) , int(intervel_y * depth + node_radius/2)), 
                        font_parm, 0.8, (0,  0, 190), 1)
    
    cv2.imshow('inference_demo', img_show)
    cv2.waitKey()
#    cv2.imwrite('E:/ncrp_infer/most_like_tree_alpha'+str(ALPHA_BASE)+'_gamma['+str(GAMMA)+']_'+str(img_num)+'.png', img_show)

    cv2.destroyAllWindows()

#def cal_p_value(f, independent_value):
#    f = Function('f')
#    return float(f.subs({'theta': independent_value[0], 'alpha': independent_value[1]}))

def main():
    loggamma_hyper_sample = np.arange(-2.3, 2.31, 0.55)
    gamma_hyper_sample = map(lambda x: np.e**x, loggamma_hyper_sample)
#    gamma_hyper_sample = np.arange(0.1, 1.01, 0.4)
    logalpha_hyper_sample = np.arange(0.4, 4.01, 0.2)
    alpha_hyper_sample = map(lambda x: np.e**x, logalpha_hyper_sample)
#    alpha_hyper_sample = np.arange(1.5, 3.51, 0.5)
    GAM, ALP = np.meshgrid(gamma_hyper_sample, alpha_hyper_sample)
    
#    for i in SUBJECT_NUM_LIST:
#        IMG_LIST.append(str(i) + '_0')
#        IMG_LIST.append(str(i) + '_1')
#    print IMG_LIST
    Posterior_Likelihood = posterior(img_num = len(IMG_LIST))
    F_Img_Likelihood = Posterior_Likelihood()
    information_content_array = np.zeros([len(F_Img_Likelihood), len(gamma_hyper_sample) * len(alpha_hyper_sample)])
    
    for i in range(len(F_Img_Likelihood)):
        information_content_array[i] = [-np.log2(float(F_Img_Likelihood[i].subs({'theta': gamma, 'alpha': alpha}))) for gamma, alpha in zip(np.ravel(GAM), np.ravel(ALP))]
    
    
    corr_list = map(lambda x: pow(np.corrcoef(information_content_array[:, x], Z_SCORE)[0][1], 2), range(len(information_content_array[0])))
    corr = np.array(corr_list)
#    partial_corr_list = map(lambda x: pow(pc.partial_corr(np.concatenate((information_content_array[:, x], Z_SCORE, SIZE_STD)).reshape(20, 3).T)[0][1], 2), range(len(information_content_array[0])))
#    partial_corr_array = np.array(partial_corr_list)
#    IC_CE = {'information_content': information_content_list}    
#    export_IC_CE = pd.DataFrame(IC_CE, columns = ['information_content'])
#    export_IC_CE.to_csv('E:/pic_IC_and_CE/informationcontent_conditionalentropy_' + str(min(gamma_hyper_sample)) + '_' + str(max(gamma_hyper_sample)) + '_' + str(min(alpha_hyper_sample)) + '_' + str(max(alpha_hyper_sample)) + '.csv')
#    print 'okkkkk'
    CORR = corr.reshape(GAM.shape)
#    PARTIAL_CORR_ARRAY = partial_corr_array.reshape(GAM.shape)
    est = zip(np.ravel(GAM), np.ravel(ALP))[np.argmax(CORR)]
#    est = zip(np.ravel(GAM), np.ravel(ALP))[np.argmax(PARTIAL_CORR_ARRAY)]
    print est
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(GAM, ALP, CORR)
#    ax.plot_surface(GAM, ALP, PARTIAL_CORR_ARRAY)
    ax.set_xlabel('GAM')
    ax.set_ylabel('ALP')
#    ax.set_zlabel('PARTIAL_CORR')
    ax.set_zlabel('CORR')
    plt.show()
    print 'okkkkk'
    
if __name__ == "__main__":
    main()
    

