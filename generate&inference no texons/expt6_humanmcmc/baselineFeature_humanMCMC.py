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
SUB_NUM = list(range(1, 17))
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

#INFORMATION_CONTENT_VALUE_LIST = [2.44, -2.48, 1.09, 4.40, -1.89, 7.29, 1.63, -0.10, 5.07, 6.81, 8.10, 7.65, 0.56, 10.5, -1.24, 3.23, 9.84, 7.12, -0.70, 5.91]
#CONDITIONAL_ENTROPY_VALUE_LIST = [1.52, 3.57,  2.63, 1.68,  3.58, 0.80, 2.31,  2.79, 1.90, 0.84, 0.76, 0.59, 2.20, 0.00, 3.67,  1.54, 0.00, 0.94, 2.30,  0.98]
INFORMATION_CONTENT_VALUE_LIST = []
CONDITIONAL_ENTROPY_VALUE_LIST = []
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

class likelihoodBaseOnFeatureValues():
    def __init__(self, img_list, sub_num):
        self.img_list = img_list
        self.sub_num = sub_num
        
    def __call__(self):
	featureValuesAllImg = pd.read_csv('featureValueAllImg.csv', index_col = 0)
        heightValuesAllImg = featureValuesAllImg[['height1', 'height2', 'height3', 'height4', 'height5', 'height6']]
	widthValuesAllImg = featureValuesAllImg[['width1', 'width2', 'width3', 'width4', 'width5', 'width6']]
	sizeValuesAllImg = featureValuesAllImg[['heightByWidth1', 'heightByWidth2', 'heightByWidth3', 'heightByWidth4', 'heightByWidth5', 'heightByWidth6']]
        meanHeightAllImg = np.mean(heightValuesAllImg.values[:], axis = 0)
        covHeightAllImg = np.cov(heightValuesAllImg.values[:].T)
        meanWidthAllImg = np.mean(widthValuesAllImg.values[:], axis = 0)
        covWidthAllImg = np.cov(widthValuesAllImg.values[:].T)
        meanSizeAllImg = np.mean(sizeValuesAllImg.values[:], axis = 0)
        covSizeAllImg = np.cov(sizeValuesAllImg.values[:].T)
        for i in self.sub_num:
            for j in self.img_list:
                img_num = str(i) + '_' + str(j)
                print(img_num)
                Possible_Tree = possible_tree_generator(img_num)
                Possible_Tree_List = Possible_Tree()
                treeExample = Possible_Tree_List[0]
                wholeImgHeight, wholeImgWidth, channel_num = treeExample.node[1]['img'].shape
                leaf_nodes = [ n for n,d in dict(treeExample.out_degree()).items() if d==0]
                partsNormalizedHeights = [treeExample.node[nodeIndex]['img'].shape[0]/wholeImgHeight for nodeIndex in leaf_nodes]
                partsNormalizedWidthes = [treeExample.node[nodeIndex]['img'].shape[1]/wholeImgWidth for nodeIndex in leaf_nodes]
                partsNormalizedSizes = [height * width for height, width in zip(partsNormalizedHeights, partsNormalizedWidthes)]
                imgFeatureVector = np.concatenate([partsNormalizedHeights, partsNormalizedWidthes, partsNormalizedSizes])
                probabilityImg = scipy.stats.multivariate_normal.pdf(partsNormalizedHeights, meanHeightAllImg, covHeightAllImg, allow_singular = True) \
                                * scipy.stats.multivariate_normal.pdf(partsNormalizedWidthes, meanWidthAllImg, covWidthAllImg, allow_singular = True) \
                                * scipy.stats.multivariate_normal.pdf(partsNormalizedSizes, meanSizeAllImg, covSizeAllImg, allow_singular = True)
                informationContentImg = -math.log(probabilityImg,2)
                INFORMATION_CONTENT_VALUE_LIST.append(informationContentImg)

def main():           
    LikelihoodBaseOnFeatureValues = likelihoodBaseOnFeatureValues(img_list = IMG_LIST, sub_num = SUB_NUM)
    LikelihoodBaseOnFeatureValues()
    IC_CE = {'information_content': INFORMATION_CONTENT_VALUE_LIST}   
    dfIndex = pd.MultiIndex.from_product([SUB_NUM, IMG_LIST], names = ['sub', 'img'])
    export_IC_CE = pd.DataFrame(IC_CE, columns = ['information_content'], index = dfIndex)
    
    fig = plt.figure()
    axForDraw = fig.add_subplot(111)
    for key, grp in export_IC_CE.groupby('img'):
        grp.index = grp.index.droplevel('img')
        grp.plot(ax=axForDraw, label=key, title='information_content', y = 'information_content')
    plt.show()
    
    export_IC_CE.to_csv('informationcontent_baseline_feature_expt6' + str(max(SUB_NUM)) + '.csv')
    print ('okkkkk')
    
if __name__ == "__main__":
    main()
    

