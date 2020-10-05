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
#import partial_corr as pc
from scipy.stats import norm as ssn

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
        img_original = cv2.imread('../scenes/' + str(10001 + IMG_LIST[self.img_num]) + '.png')
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

#def cal_possible_cut_edges(img): 
#    img_height, img_width, channel_num = img.shape
#    edges_vertical = [-1]
#    edges_horizontal = [-1]
#    for i in range(4, img_width - 3):
#        if (img[0][i][0] != img[0][i + 1][0] or img[0][i][1] != img[0][i + 1][1] or img[0][i][2] != img[0][i + 1][2]):  
#            for k in range(i-2, i+3):
#                if (img[img_height - 1][k][0] != img[img_height - 1][k + 1][0] or img[img_height - 1][k][1] != img[img_height - 1][k + 1][1] or img[img_height - 1][k][2] != img[img_height - 1][k + 1][2]):
#                    edges_vertical.append(i)
#                    break
#    edges_vertical.append(img_width - 1)
#    
#    for j in range(4, img_height - 3):
#        if (img[j][0][0] != img[j + 1][0][0] or img[j][0][1] != img[j + 1][0][1] or img[j][0][2] != img[j + 1][0][2]):
#            for l in range(j-2, j+3):
#                if (img[l][img_width - 1][0] != img[l + 1][img_width - 1][0] or img[l][img_width - 1][1] != img[l + 1][img_width - 1][1] or img[l][img_width - 1][2] != img[l + 1][img_width - 1][2]):   
#                    edges_horizontal.append(j)
#                    break
#    edges_horizontal.append(img_height - 1)
#
#    return edges_vertical, edges_horizontal
 
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


"""cal the posterior probability and visualization the most likely tree"""
class likelihoodBaseOnFeatureValues():
    def __init__(self, img_num):
        self.img_num = img_num
        
    def __call__(self):
#        img_loglikelihood_given_gamma_alpha = 1
#        f_img_likelihood = Function('f_img_likelihood')
#        f_all_img = []
	featureValuesAllImg = pd.read_csv('featureValueAllImg.csv', index_col = 0)
        featureValuesTrainImgOrigin = featureValuesAllImg
        heightValuesTrainImgOrigin = featureValuesTrainImgOrigin[['height1', 'height2', 'height3', 'height4', 'height5']]#, 'height6']]
        widthValuesTrainImgOrigin = featureValuesTrainImgOrigin[['width1', 'width2', 'width3', 'width4', 'width5']]#, 'width6']]
        sizeValuesTrainImgOrigin = featureValuesTrainImgOrigin[['heightByWidth1', 'heightByWidth2', 'heightByWidth3', 'heightByWidth4', 'heightByWidth5']]#, 'heightByWidth6']]
        
        numTrain = len(featureValuesTrainImgOrigin)
        print numTrain
        nodeIndex = np.array([np.arange(0, 5, 1) for _ in range(numTrain)])
        map(np.random.shuffle, nodeIndex)

        heightValuesTrainImg = np.array([heightValuesTrainImgOrigin.values[i][nodeIndex[i]] for i in range(numTrain)])
        widthValuesTrainImg = np.array([widthValuesTrainImgOrigin.values[i][nodeIndex[i]] for i in range(numTrain)])
        sizeValuesTrainImg = np.array([sizeValuesTrainImgOrigin.values[i][nodeIndex[i]] for i in range(numTrain)])

        meanHeightTrainImg = np.mean(heightValuesTrainImg, axis = 0)
        #covHeightTrainImg = np.var(heightValuesTrainImg, axis = 0) 
        stdHeightTrainImg = np.std(heightValuesTrainImg, axis = 0) 
        meanWidthTrainImg = np.mean(widthValuesTrainImg, axis = 0)
        #covWidthTrainImg = np.var(widthValuesTrainImg, axis = 0) #/ (numTrain - 1) * numTrain
        stdWidthTrainImg = np.std(widthValuesTrainImg, axis = 0) #/ (numTrain - 1) * numTrain
        meanSizeTrainImg = np.mean(sizeValuesTrainImg, axis = 0)
        #covSizeTrainImg = np.var(sizeValuesTrainImg, axis = 0) #/ (numTrain - 1) * numTrain
        stdSizeTrainImg = np.std(sizeValuesTrainImg, axis = 0) #/ (numTrain - 1) * numTrain
        #print meanHeightTrainImg, stdWidthTrainImg
        #print meanSizeTrainImg, stdSizeTrainImg
        model_p_list = [0] * 20
        for z in range(self.img_num):
            Possible_Tree = possible_tree_generator(z)
            Possible_Tree_List = Possible_Tree()
            treeExample = Possible_Tree_List[0]
            wholeImgHeight, wholeImgWidth, channel_num = treeExample.node[1]['img'].shape
            leaf_nodes = [ n for n,d in dict(treeExample.out_degree()).items() if d==0]
            partsHeights = np.array([treeExample.node[nodeIndex]['img'].shape[0] / wholeImgHeight for nodeIndex in leaf_nodes][0:5])
            partsWidthes = np.array([treeExample.node[nodeIndex]['img'].shape[1] / wholeImgWidth for nodeIndex in leaf_nodes][0:5])
            partsSizes = np.array([height * width for height, width in zip(partsHeights, partsWidthes)][0:5])
            partsNormalizedHeights = (partsHeights - meanHeightTrainImg) / stdHeightTrainImg 
            partsNormalizedWidthes = (partsWidthes - meanWidthTrainImg) / stdWidthTrainImg 
            partsNormalizedSizes = (partsSizes - meanSizeTrainImg) / stdSizeTrainImg 
            imgFeatureVector = np.concatenate([partsNormalizedHeights, partsNormalizedWidthes, partsNormalizedSizes])
	    model_p_list[z] = scipy.stats.multivariate_normal.pdf(partsNormalizedHeights, np.zeros(5), np.ones(5), allow_singular = True) \
                                * scipy.stats.multivariate_normal.pdf(partsNormalizedWidthes, np.zeros(5), np.ones(5), allow_singular = True) \
                                * scipy.stats.multivariate_normal.pdf(partsNormalizedSizes, np.zeros(5), np.ones(5), allow_singular = True)
	print(model_p_list)
        return model_p_list

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
#    loggamma_hyper_sample = np.arange(-2.3, 2.31, 0.55)
#    gamma_hyper_sample = map(lambda x: np.e**x, loggamma_hyper_sample)
##    gamma_hyper_sample = np.arange(0.1, 1.01, 0.4)
#    logalpha_hyper_sample = np.arange(0.4, 4.01, 0.2)
#    alpha_hyper_sample = map(lambda x: np.e**x, logalpha_hyper_sample)
##    alpha_hyper_sample = np.arange(1.5, 3.51, 0.5)
#    GAM, ALP = np.meshgrid(gamma_hyper_sample, alpha_hyper_sample)
    
#    for i in SUBJECT_NUM_LIST:
#        IMG_LIST.append(str(i) + '_0')
#        IMG_LIST.append(str(i) + '_1')
#    print IMG_LIST
    LikelihoodBaseOnFeatureValues = likelihoodBaseOnFeatureValues(img_num = len(IMG_LIST))
    Model_P_List = LikelihoodBaseOnFeatureValues()
    
    model_z_score = np.zeros(20)
    for m in range(12):
        print 'm', m
        model_choice_list = np.zeros(20)
        for i in range(20):
            for j in range(20):
                if i != j:
                    p = [Model_P_List[i], Model_P_List[j]]
                    p_normalize = np.array(p) * 1.0 / np.sum(p)
                    i_j_index = list(np.random.multinomial(1, p_normalize)).index(1)
                    if i_j_index == 0:
                        model_choice_list[i] = model_choice_list[i] + 1
                    else:
                        model_choice_list[j] = model_choice_list[j] + 1
        choice_p_array = (np.array(model_choice_list) + 1)/ 40
        z_value_array = map(lambda x: ssn.ppf(x), choice_p_array) 
        model_z_score =  model_z_score + z_value_array
    corr_model_human = pow(np.corrcoef(model_z_score, Z_SCORE)[0][1], 2)
    print corr_model_human
#    information_content_array = np.zeros([len(F_Img_Likelihood), len(gamma_hyper_sample) * len(alpha_hyper_sample)])
#    
#    for i in range(len(F_Img_Likelihood)):
#        information_content_array[i] = [-np.log2(float(F_Img_Likelihood[i].subs({'theta': gamma, 'alpha': alpha}))) for gamma, alpha in zip(np.ravel(GAM), np.ravel(ALP))]
#    
#    
#    corr_list = map(lambda x: pow(np.corrcoef(information_content_array[:, x], Z_SCORE)[0][1], 2), range(len(information_content_array[0])))
#    corr = np.array(corr_list)
##    partial_corr_list = map(lambda x: pow(pc.partial_corr(np.concatenate((information_content_array[:, x], Z_SCORE, SIZE_STD)).reshape(20, 3).T)[0][1], 2), range(len(information_content_array[0])))
##    partial_corr_array = np.array(partial_corr_list)
    IC_CE = {'model_z_score': model_z_score}    
    export_IC_CE = pd.DataFrame(IC_CE, columns = ['model_z_score'])
    export_IC_CE.to_csv('model_z_score_baseline_feature.csv')
##    print 'okkkkk'
#    CORR = corr.reshape(GAM.shape)
##    PARTIAL_CORR_ARRAY = partial_corr_array.reshape(GAM.shape)
#    est = zip(np.ravel(GAM), np.ravel(ALP))[np.argmax(CORR)]
##    est = zip(np.ravel(GAM), np.ravel(ALP))[np.argmax(PARTIAL_CORR_ARRAY)]
#    print est
#    
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')
#    ax.plot_surface(GAM, ALP, CORR)
##    ax.plot_surface(GAM, ALP, PARTIAL_CORR_ARRAY)
#    ax.set_xlabel('GAM')
#    ax.set_ylabel('ALP')
##    ax.set_zlabel('PARTIAL_CORR')
#    ax.set_zlabel('CORR')
#    plt.show()
    print 'okkkkk'
    
if __name__ == "__main__":
    main()
    

