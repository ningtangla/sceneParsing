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

sys.setrecursionlimit(1000000000)
"""
number of img to infer
"""
SUBJECT_NUM_LIST = range(1, 26) + range(101, 104) + range(105, 111) + range(201, 209) + range(301, 311)
IMG_LIST = []

IMG_NUM_BATCH = 2
IMG_NUM = 10001

"""
ncrp parameters
"""

GAMMA = 0
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
ALPHA_BASE = 0

""" possible tree and its p value for curr img """
    
class possible_tree_generator():  
    def __init__(self, img_num):
        self.img_num = img_num
        self.all_img_tree_list = []
        
    def __call__(self):
        img_original = cv2.imread('draw/blank1.0/'+IMG_LIST[self.img_num]+'.png')
        img_init = img_original[153:615, 204:820]
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
    
class measureTerminalPartsFeatures():
    def __init__(self, img_num):
        self.img_num = img_num
        
    def __call__(self):
        featureValueAllImg = []
        for z in range(self.img_num):
            print z
            Possible_Tree = possible_tree_generator(z)
            Possible_Tree_List = Possible_Tree()
            treeExample = Possible_Tree_List[0]
            wholeImgHeight, wholeImgWidth, channel_num = treeExample.node[1]['img'].shape
            leaf_nodes = [ n for n,d in dict(treeExample.out_degree()).items() if d==0]
            partsNormalizedHeights = [treeExample.node[nodeIndex]['img'].shape[0]/wholeImgHeight for nodeIndex in leaf_nodes]
            partsNormalizedWidthes = [treeExample.node[nodeIndex]['img'].shape[1]/wholeImgWidth for nodeIndex in leaf_nodes]
            partsNormalizedSizes = [height * width for height, width in zip(partsNormalizedHeights, partsNormalizedWidthes)]
            imgFeatureVector = np.concatenate([partsNormalizedHeights, partsNormalizedWidthes, partsNormalizedSizes])
            featureValueAllImg.append(imgFeatureVector)
        featureDataFrame = pd.DataFrame(featureValueAllImg, columns = ['height1', 'height2', 'height3', 'height4', 'height5', 'height6', 'width1', 'width2', 'width3', 'width4', 'width5', 'width6', 'heightByWidth1', 'heightByWidth2', 'heightByWidth3', 'heightByWidth4', 'heightByWidth5', 'heightByWidth6'])
        featureDataFrame.to_csv('featureValueAllImg.csv')
        return featureValueAllImg


def main():
    for i in SUBJECT_NUM_LIST:
        IMG_LIST.append(str(i) + '_0')
        IMG_LIST.append(str(i) + '_1')
    MeasureTerminalPartsFeatures = measureTerminalPartsFeatures(len(IMG_LIST))

    featureValueAllImg = MeasureTerminalPartsFeatures()
    print 'okkkkk'
    
if __name__ == "__main__":
    main()
    

