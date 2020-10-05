# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 13:52:00 2017

output: no all_possible_infer_trees, only most_likely_tree

@author: Edward Coen
"""

from __future__ import division
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

sys.setrecursionlimit(1000000000)
"""
number of img to infer
"""

IMG_LIST = list(np.array([5, 18, 42, 77, 157, 168, 183, 369, 407, 447, 587, 619, \
        632, 653, 660, 702, 732, 750, 842, 864]) + 10000)
print(IMG_LIST)
SUB_NUM = list(range(0, 17))
"""
ncrp parameters
"""

GAMMA = 0.9
ALL_GUEST_NUM = 6

"""
image parameters
"""

#IMAGE_WIDTH = 1024
#IMAGE_HEIGHT = 768
COLOR_SPACE = [[128,128,128],[0,0,255],[0,255,0],[255,0,0],[0,255,255],[255,0,255],[255,255,0],[255,255,255]]
GRAPH_WIDTH = 500
TREE_GRAPH_HEIGHT = 252
INFER_IMG_ADJUST_HEIGHT = 300

"""
Dirchlet parmeters
"""
ALPHA_BASE = [3.5]

"""
information content record
"""

INFORMATION_CONTENT_VALUE_LIST = []
CONDITIONAL_ENTROPY_VALUE_LIST = []
NUM_TREE_SPACE_LIST = []
DEPTH_FLATTEST_TREE_List = []
"""
code parameters
"""

"""
global arguements
"""

    
""" possible tree and its p value for curr img """
    
class possible_tree_generator():  
    def __init__(self, img_num):
        self.img_num = img_num
        self.all_img_tree_list = []
        
    def __call__(self):
        img_origin = cv2.imread('image/'+str(self.img_num)+'.png')
        img_height, img_width, channel_num = img_origin.shape
        if self.img_num[0] != str(0):
            img_init = img_origin[int(0.2 * img_height) : int(0.8 * img_height), int(0.2 * img_width) : int(0.8 * img_width)]
        else:
            img_init = img_origin
            print(self.img_num)
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
        leaf_nodes = [ n for n,d in dict(tree.out_degree()).items() if d==0]
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
                    tree_temp.node[1]['p_like'] = tree_temp.node[1]['p_like']*cal_p_dirichlet(cut_propotion)*0.5
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
                    tree_temp.node[1]['p_like'] = tree_temp.node[1]['p_like']*cal_p_dirichlet(cut_propotion)*0.5
                    extended_trees.append(tree_temp)              
        return extended_trees
 
def cal_possible_cut_edges(img): 
    img_height, img_width, channel_num = img.shape
    edges_vertical = [-1]
    edges_horizontal = [-1]
    for i in range(5, img_width - 4):
        if (img[5][i][2] == 0 and img[5][i + 1][2] == 255):  
            for k in range(i-2, i+3):
                if (img[img_height - 5][k][2] == 0 and img[img_height - 5][k + 1][2] == 255):
                    edges_vertical.append(i)
                    break
    edges_vertical.append(img_width - 1)
    
    for j in range(5, img_height - 4):
        if (img[j][5][2] == 0 and img[j + 1][5][2] == 255):
            for l in range(j-2, j+3):
                if (img[l][img_width - 5][2] == 0 and img[l + 1][img_width - 5][2] == 255):   
                    edges_horizontal.append(j)
                    break
    edges_horizontal.append(img_height - 1)

    return edges_vertical, edges_horizontal

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
    nonleaf_nodes = [ n for n,d in dict(tree.out_degree()).items() if d!=0]
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
            
        children = list(nx.DiGraph.successors(self.graph, node_id))
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
        return reduce(op.mul, np.array(probability_table_partion)*np.array(all_combination_guest)*np.array(renomalize_parameter))
        
    def __call__(self):
        return map(self.cal_prior_probability, self.all_table_partion_list)
    
def getTreeDepth(tree):
    leafNodes = [ n for n,d in dict(tree.out_degree()).items() if d==0]
    depth = max([tree.node[leafNode]['depth'] for leafNode in leafNodes])
    return depth

"""cal the posterior probability and visualization the most likely tree"""
class posterior():
    def __init__(self, img_list, sub_num):
        self.img_list = img_list
        self.sub_num = sub_num
        
    def __call__(self):
        for i in self.sub_num:
            for j in self.img_list:
                img_num = str(i) + '_' + str(j)
                print(img_num)
                Possible_Tree = possible_tree_generator(img_num)
                Possible_Tree_List = Possible_Tree()
                Likelihood = likelihood(all_tree_graph = Possible_Tree_List)
                Likelihood_Probability = Likelihood()
                All_Table_Partion_List = map(lambda x: Likelihood_Probability[0][x], range(len(Likelihood_Probability[0])))
                Prior = prior(all_table_partion_list = All_Table_Partion_List)
                Prior_Probability =  Prior()
                posterior_list = map(lambda x: Prior_Probability[x] * Likelihood_Probability[1][x], range(len(Likelihood_Probability[0])))
                max_index = posterior_list.index(np.array(posterior_list).max())
                most_like_tree = Possible_Tree_List[max_index]
                most_like_tree.node[1]['possible_trees_num'] = len(Possible_Tree_List)
                most_like_tree.node[1]['p_poster_list'] = posterior_list
                
                numTreeSpace = len(Possible_Tree_List)
                NUM_TREE_SPACE_LIST.append(numTreeSpace)
                
                treeDepthes = [getTreeDepth(tree) for tree in Possible_Tree_List]
                flattestTreeDepth = min(treeDepthes)
                DEPTH_FLATTEST_TREE_List.append(flattestTreeDepth)
                
                information_content = - math.log(np.array(posterior_list).sum(), 2)
                INFORMATION_CONTENT_VALUE_LIST.append(information_content)
                
                posterior_list_renormalize = np.array(posterior_list) / np.array(posterior_list).sum()
                conditional_entropy_list = map(lambda x: - x * math.log(x), posterior_list_renormalize)
                conditional_entropy = np.array(conditional_entropy_list).sum()
                CONDITIONAL_ENTROPY_VALUE_LIST.append(conditional_entropy) 

def main():           
    Posterior_Likelihood = posterior(img_list = IMG_LIST, sub_num = SUB_NUM)
    Most_Like_Tree = Posterior_Likelihood()
    IC_CE = {'information_content': INFORMATION_CONTENT_VALUE_LIST, 'conditional_entropy': CONDITIONAL_ENTROPY_VALUE_LIST, 'num_TreeSpace': NUM_TREE_SPACE_LIST,
            'flattestTreeDepth': DEPTH_FLATTEST_TREE_List}

    dfIndex = pd.MultiIndex.from_product([SUB_NUM, IMG_LIST], names = ['sub', 'img'])
    export_IC_CE = pd.DataFrame(IC_CE, columns = ['information_content', 'conditional_entropy', 'num_TreeSpace', 'flattestTreeDepth'], index = dfIndex)
    
    fig = plt.figure()
    axForDraw = fig.add_subplot(111)
    for key, grp in export_IC_CE.groupby('img'):
        grp.index = grp.index.droplevel('img')
        grp.plot(ax=axForDraw, label=key, title='information_content', y = 'information_content')
    plt.show()
    
    export_IC_CE.to_csv('image/new_informationcontent_conditionalentropy_numTreeSpace_depthFlattestTree_expt6_sub' + str(max(SUB_NUM)) + '.csv')
    print ('okkkkk')
    
if __name__ == "__main__":
    main()
    

