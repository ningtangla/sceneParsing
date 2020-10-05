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
import matplotlib.ticker as ticker
import graphviz
import cPickle as pickle
from sympy import *

sys.setrecursionlimit(1000000000)
"""
number of img to infer
"""
SUBJECT_NUM_LIST = range(1, 26) + range(101, 104) + range(105, 111) + range(201, 209) + range(301, 311)
IMG_LIST = np.array([1, 4, 6, 9, 10, 13, 14, 16, 22, 26, 29, 31, 33, 39, 40, 47, 48, 54, 60, 61, 63, 72, 73, 78, 79, 80, 81, 84, 86, 87, 90, 94, 97, 103, 111, 117, 134, 136, 142, 148, 164, 187, 188, 228, 229, 275, 613, 614]) + 10000
#HUMAN_LIST = np.array([23, 999, 14, 44, 26, 999, 28, 3, 13, 999, 9, 8, 26, 999, 44, 25, 40, 999, 999, 999, 7, 999, 9, 999, 30, 44, 999, 999, 999, 28, 44, 999, 16, 32, 30, 14, 31, 26, 999, 32, 28, 10, 999, 17, 42, 99, 6, 40])
HUMAN_LIST = np.array([999] * 48)
print len(HUMAN_LIST)
IMG_LIST = np.array([28004])
IMG_NUM_BATCH = 1
IMG_NUM = 10001

"""
ncrp parameters
"""

GAMMA = 1
GAMMAList = [0.1, 1, 10]
ALL_GUEST_NUM = 5

"""
image parameters
"""

IMAGE_WIDTH = 512
IMAGE_HEIGHT = 768
COLOR_SPACE = [[128,128,128],[0,0,255],[0,255,0],[255,0,0],[0,255,255],[255,0,255],[255,255,0],[255,255,255]]
GRAPH_WIDTH = 500
TREE_GRAPH_HEIGHT = 252
INFER_IMG_ADJUST_HEIGHT = 300

"""
Dirchlet parmeters
"""
ALPHA_BASE = 3.5


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
        img_original = cv2.imread('demo_nocolor/'+str(self.img_num)+'.png')
#        img_init = img_original[153:615, 204:820]
        img_init = img_original
#        cv2.namedWindow('image')
#        cv2.imshow('image', img_init)
#        cv2.waitKey()
#        cv2.destroyAllWindows()
        tree_init = nx.DiGraph()
        tree_init.add_node(1, img = img_init, complete = 0, terminal = 0, x = [1, 1024], y = [1, 768], direction = [], p_like = 1, p_poster_list = [], guest = [], depth = 1, possible_trees_num = 0)
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
        last_x = tree.node[curr_leaf_node]['x']
        last_y = tree.node[curr_leaf_node]['y']
        if len(edges_vertical) != 2:
            tree.node[curr_leaf_node]['direction'] = 0
            for i in range(len(edges_vertical) - 2):
                combination = list(itertools.combinations(range(len(edges_vertical) - 2), i + 1))
                combination_list = map(list, combination)
                
                for z in range(len(combination_list)):
                    tree_temp = tree.copy()
                    cut_propotion = []
                    
                    tree_temp.add_node(new_node_id_start + 1, img = img_to_cut[0:img_height, edges_vertical[0] + 1:edges_vertical[combination_list[z][0] + 1]], terminal = 0, x = [last_x[0], last_x[0] + edges_vertical[combination_list[z][0] + 1]], y = last_y, direction = [], cut_proportion = [], guest = [], depth = tree.node[curr_leaf_node]['depth'] + 1)
                    tree_temp.add_edge(curr_leaf_node , new_node_id_start + 1)
                    cut_propotion.append((edges_vertical[combination_list[z][0] + 1] - (edges_vertical[0] + 1) + 1)/img_width)
                    if len(combination_list[z]) != 1:
                        for j in range(len(combination_list[z]) - 1):
                            tree_temp.add_node(new_node_id_start + 1 + (j + 1), img = img_to_cut[0:img_height, (edges_vertical[combination_list[z][j] + 1] + 1):edges_vertical[combination_list[z][j + 1] + 1]], terminal = 0, x = [last_x[0] + edges_vertical[combination_list[z][j] + 1], last_x[0] + edges_vertical[combination_list[z][j + 1] + 1]], y = last_y, direction = [], cut_proportion = [], guest = [], depth = tree.node[curr_leaf_node]['depth'] + 1)
                            tree_temp.add_edge(curr_leaf_node, new_node_id_start + 1 + (j + 1))
                            cut_propotion.append((edges_vertical[combination_list[z][j + 1] + 1] - (edges_vertical[combination_list[z][j] + 1] + 1) + 1)/img_width) 
                            
                    tree_temp.add_node(new_node_id_start + len(combination_list[z]) + 1, img = img_to_cut[0:img_height, edges_vertical[combination_list[z][-1] + 1] + 1:edges_vertical[-1]], terminal = 0, x = [last_x[0] + edges_vertical[combination_list[z][-1] + 1], last_x[1]], y = last_y, direction = [], cut_proportion = [], guest = [], depth = tree.node[curr_leaf_node]['depth'] + 1)
                    tree_temp.add_edge(curr_leaf_node , new_node_id_start + len(combination_list[z]) + 1)
                    cut_propotion.append(1 - np.array(cut_propotion).sum())
                    
                    tree_temp.node[curr_leaf_node]['cut_proportion'] = cut_propotion
                    tree_temp.node[1]['p_like'] = tree_temp.node[1]['p_like']*cal_p_dirichlet(cut_propotion)*0.5
                    extended_trees.append(tree_temp)
                    
        else:
            tree.node[curr_leaf_node]['direction'] = 1
            for i in range(len(edges_horizontal) - 2):
                combination = list(itertools.combinations(range(len(edges_horizontal) - 2), i + 1))
                combination_list = map(list, combination)
                for z in range(len(combination_list)):
                    tree_temp = tree.copy()
                    cut_propotion = []
                    
                    tree_temp.add_node(new_node_id_start + 1, img = img_to_cut[edges_vertical[0] + 1:edges_horizontal[combination_list[z][0] + 1], 0:img_width], terminal = 0, y = [last_y[0], last_y[0] + edges_horizontal[combination_list[z][0] + 1]], x = last_x, direction = [], cut_proportion = [], guest = [], depth = tree.node[curr_leaf_node]['depth'] + 1)
                    tree_temp.add_edge(curr_leaf_node , new_node_id_start + 1)
                    cut_propotion.append((edges_horizontal[combination_list[z][0] + 1] - (edges_horizontal[0] + 1) + 1)/img_height)
                    if len(combination_list[z]) != 1:
                        for j in range(len(combination_list[z]) - 1):
                            tree_temp.add_node(new_node_id_start + 1 + (j + 1), img = img_to_cut[(edges_horizontal[combination_list[z][j] + 1] + 1):edges_horizontal[combination_list[z][j + 1] + 1], 0:img_width], terminal = 0, y = [last_y[0] + edges_horizontal[combination_list[z][j] + 1], last_y[0] + edges_horizontal[combination_list[z][j + 1] + 1]], x = last_x, direction = [], cut_proportion = [], guest = [], depth = tree.node[curr_leaf_node]['depth'] + 1)
                            tree_temp.add_edge(curr_leaf_node, new_node_id_start + + 1 + (j + 1))
                            cut_propotion.append((edges_horizontal[combination_list[z][j + 1] + 1] - (edges_horizontal[combination_list[z][j] + 1] + 1) + 1)/img_height)       
                    
                    tree_temp.add_node(new_node_id_start + len(combination_list[z]) + 1, img = img_to_cut[edges_horizontal[combination_list[z][-1] + 1] + 1:edges_horizontal[-1], 0:img_width], terminal = 0, y = [last_y[0] + edges_horizontal[combination_list[z][-1] + 1], last_y[1]], x = last_x, direction = [], cut_proportion = [], guest = [], depth = tree.node[curr_leaf_node]['depth'] + 1)
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
    alpha = [ALPHA_BASE] * len(cut_propotion)
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

def getTreeSortStandard(tree):
#    leaf_nodes = [n for n,d in dict(tree.out_degree()).items() if d==0]
    nodeDepthCount = np.zeros(ALL_GUEST_NUM)
    for nodeID in tree.nodes():
       depth = tree.node[nodeID]['depth']
       nodeDepthCount[depth - 1] += 1

    nonleaf_nodes = [n for n,d in dict(tree.out_degree()).items() if d!=0]
    nonleafDepthCount = np.zeros(ALL_GUEST_NUM)
    for nodeID in nonleaf_nodes:
       depth = tree.node[nodeID]['depth']
       nonleafDepthCount[depth - 1] += 1
    
    maxDepth =np.max(np.nonzero(nodeDepthCount))

    branchingFactor = [nodeDepthCount[depthId] / nonleafDepthCount[depthId - 1] for depthId in range(1, maxDepth + 1)]
    bfScoreEachLayer = [round(10 * branchingFactor[depthId]) * 10 * np.power(1000, ALL_GUEST_NUM - 1 - depthId) for depthId in range(maxDepth)]
    leafNumScoreEachLayer = [int(nodeDepthCount[depthId] - nonleafDepthCount[depthId]) * np.power(1000, ALL_GUEST_NUM - 1 - depthId) for depthId in range(maxDepth)]
    #nonLeafNumScoreEachLayer = [int(nonleafDepthCount[depthId]) * np.power(1000, ALL_GUEST_NUM - 1 - depthId) for depthId in range(1, maxDepth + 1)]
    treeCode = np.sum(np.array(bfScoreEachLayer) + np.array(leafNumScoreEachLayer))
    return maxDepth, treeCode 

def drawPerformanceLine(dataDf, axForDraw, title):
    dataDf.plot(ax=axForDraw, label=title, y='p')
    axForDraw.xaxis.set_major_locator(ticker.MultipleLocator(1))

def distributeTree(conditionDf):
    global GAMMA
    GAMMA = conditionDf.index.get_level_values('gamma')[0]
    possible_tree_num = []
    
    imgNum = IMG_LIST[0]

    Possible_Tree = possible_tree_generator(imgNum)
    Possible_Tree_List = Possible_Tree()
    Likelihood = likelihood(all_tree_graph = Possible_Tree_List)
    Likelihood_Probability = Likelihood()
    All_Table_Partion_List = map(lambda x: Likelihood_Probability[0][x], range(len(Likelihood_Probability[0])))
    Prior = prior(all_table_partion_list = All_Table_Partion_List)
    Prior_Probability =  Prior()
    
    TreeLeafCodes = [getTreeSortStandard(tree) for tree in Possible_Tree_List]

    unrepeatedTreeLeafCodes = []
    unrepeatedTreeProbabilities = [] 
    for i in range(ALL_GUEST_NUM):
        unrepeatedTreeLeafCodes.append([])  
        unrepeatedTreeProbabilities.append([])
    for treeId in range(len(Possible_Tree_List)):
        maxDepth, treeLeafCodes = TreeLeafCodes[treeId]
        treeProbability = Prior_Probability[treeId]
        if treeLeafCodes not in unrepeatedTreeLeafCodes[maxDepth]:
            unrepeatedTreeLeafCodes[maxDepth].append(treeLeafCodes)
            unrepeatedTreeProbabilities[maxDepth].append(treeProbability)
        else:
            indexInUnrepeatedLeafCodes = list(unrepeatedTreeLeafCodes[maxDepth]).index(treeLeafCodes)
            unrepeatedTreeProbabilities[maxDepth][indexInUnrepeatedLeafCodes] += treeProbability
    
    treeIndex = []
    for depth in range(len(unrepeatedTreeProbabilities)):
        argsortIndex = np.argsort(-np.array(unrepeatedTreeLeafCodes[depth]))
        unrepeatedTreeProbabilities[depth] = np.array(unrepeatedTreeProbabilities[depth])[argsortIndex]
        unrepeatedTreeLeafCodes[depth] = np.array(unrepeatedTreeLeafCodes[depth])[argsortIndex]
        if depth!= 0:
            for treeOrder in range(len(unrepeatedTreeProbabilities[depth])):
                treeIndex.append(str(depth) + '-' + str(treeOrder))
    priorsSorted = np.concatenate(unrepeatedTreeProbabilities)
    print(np.array(unrepeatedTreeLeafCodes))
    #argsortIndex = np.argsort(treeDepthesScores)
    #priorsSorted = np.array(Prior_Probability)[np.array(argsortIndex)]
    return pd.DataFrame({'p':priorsSorted}, index = pd.Index(list(treeIndex), name = 'tree'))

def main(): 
    dfIndex = pd.MultiIndex.from_product([GAMMAList], names = ['gamma'])
    toSplitFrame = pd.DataFrame(index=dfIndex)
    distributionDf = toSplitFrame.groupby(['gamma']).apply(distributeTree)
    
    fig = plt.figure()
    plotRowNum = 1
    plotColNum = 1
    plotCounter = 1

    axForDraw = fig.add_subplot(plotRowNum, plotColNum, plotCounter)
    for (key, dataDf) in distributionDf.groupby('gamma'):
        dataDf.index = dataDf.index.droplevel('gamma')
        drawPerformanceLine(dataDf, axForDraw, str(key))
        plotCounter += 1

    plt.legend(loc='best')

    plt.show()
    
if __name__ == "__main__":
    main()
    

