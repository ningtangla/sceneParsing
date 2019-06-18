# -*- coding: utf-8 -*-
"""
Created on Sun Mar 04 14:10:20 2018

@author: Edward Coen
"""

import networkx
import operator
import itertools
import numpy
import cv2
import random
from psychopy import visual,core,event
import time
import operator as op
import networkx as nx
import scipy.special
from scipy.misc import comb
import math
import scipy.stats

GAMMA = numpy.random.uniform(0.1, 10)
ALPHA_BASE = numpy.random.uniform(0.1, 10)
ALL_GUEST_NUM = 6

def readTreeFromAnswer(pathOfAnswer, filenameOfAnswer):
    treeFromAnswer = networkx.read_gpickle(pathOfAnswer + filenameOfAnswer)
    return treeFromAnswer

def displaySceneParsingImageByTree(tree, lineMaxWidth, colorSpace):
    nonleafNodes = [nodeID for nodeID, nodeOutDegree in tree.out_degree().items() if nodeOutDegree != 0]
    imageWidth = 0.8 * numpy.max(tree.node[1]['x'])
    imageHeight = numpy.max(tree.node[1]['y'])
    image = numpy.zeros([int(imageHeight), int(imageWidth), 3], 'uint8')
    image[ int(0.8*imageHeight) : int(imageHeight), : , :] = 255
    cv2.namedWindow('image')
    cv2.imshow('image', image)

    depthOfLastNode = 1
    for currNode in nonleafNodes:
        depthOfCurrentNode = tree.node[currNode]['depth']
        childrenOfCurrNode = tree.successors(currNode)
        childrenOfCurrNode.sort()

        if tree.node[currNode]['direction'] == 0:
            linesCoordinates = map(lambda z: [(tree.node[z]['x'][1], tree.node[currNode]['y'][0]),
                                              (tree.node[z]['x'][1], tree.node[currNode]['y'][1])],
                                             childrenOfCurrNode[ :-1])
        if tree.node[currNode]['direction'] == 1:
            linesCoordinates = map(lambda z: [(tree.node[currNode]['x'][0], tree.node[z]['y'][1]),
                                              (tree.node[currNode]['x'][1], tree.node[z]['y'][1])],
                                             childrenOfCurrNode[ :-1])

        if depthOfCurrentNode > depthOfLastNode:
            cv2.waitKey()
            cv2.imshow('image', image)

        for lineCoordinate in linesCoordinates:
            cv2.line(image, (int(0.8 * lineCoordinate[0][0]), int(0.8 * lineCoordinate[0][1])), (int(0.8 * lineCoordinate[1][0]), int(0.8 * lineCoordinate[1][1])), colorSpace[tree.node[currNode]['depth'] - 1], int(lineMaxWidth * numpy.power(0.5, tree.node[currNode]['depth'])))

        depthOfLastNode = depthOfCurrentNode

    cv2.waitKey()
    questionImg = cv2.imread('question.png')
    image[int(0.9*imageHeight) - 73: int(0.9*imageHeight) + 73, int(0.5*imageWidth) - 242: int(0.5*imageWidth) + 242] = questionImg
    cv2.imshow('image', image)

    response = cv2.waitKey()
    image = numpy.zeros([int(imageHeight), int(imageWidth), 3], 'uint8')
    cv2.imshow('image', image)
    return response

class getAllTreeCRPCode():
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
        #self.all_tree_p.append(tree_graph.node[1]['p_like'])
            
    def cal_code_list(self, node_id):
            
        children = nx.DiGraph.successors(self.graph, node_id)
        if children:
            self.element_sorted_for = map(lambda x: (len(self.graph.node[x]['guest']), self.cal_children_single_guest_num(x)), children)
            self.element_no_repeat = list(set(self.element_sorted_for))
            self.element_no_repeat.sort()
            self.element_with_index_list = list(enumerate(self.element_sorted_for))
            self.element_sorted_for.sort()
            element_to_append = list(numpy.array(self.element_sorted_for)[:, 0])
            children_index_sorted = numpy.array(map(self.cal_index_sorted, self.element_no_repeat))
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
        return self.all_tree_code

"""cal the prior probability for every possible tree of curr img by the 'table partion list' form"""
    
class prior():  
    
    def __init__(self, all_table_partion_list):
        self.all_table_partion_list = all_table_partion_list
    
    def cal_renomalize_parameter(self, table_partion):
        return 1/((1 - scipy.special.gamma(GAMMA) * GAMMA * scipy.special.gamma(numpy.array(table_partion).sum()) / scipy.special.gamma(numpy.array(table_partion).sum() + GAMMA)) * len(list(set(itertools.permutations(numpy.array(table_partion)))))) 
        
    def cal_probability_table_partion(self, table_partion):
        return reduce(op.mul, map(scipy.special.gamma, numpy.array(table_partion))) * scipy.special.gamma(GAMMA) * pow(GAMMA, len(table_partion)) / scipy.special.gamma(numpy.array(table_partion).sum() + GAMMA)
            
    def cal_permutation_table_partion(self, table_partion):
        return list(set(list(itertools.permutations(table_partion))))
    
    def cal_all_combination_guest(self, permutation_table_partion): 
        return reduce(op.add, map(self.cal_permutation_combination_guest, permutation_table_partion))
        
    def cal_permutation_combination_guest(self, table_partion):
        self.guest_left = numpy.array(table_partion).sum()
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
        return reduce(op.mul, numpy.array(probability_table_partion)*numpy.array(all_combination_guest)*numpy.array(renomalize_parameter))
        
    def __call__(self):
        return map(self.cal_prior_probability, self.all_table_partion_list)
    
class experimentOfDecision():

    def __init__(self,lineMaxWidth, colorSpace):

        self.lineMaxWidth = lineMaxWidth
        self.colorSpace = colorSpace

    def __call__(self, trialTotalNum, pathOfExperiment):

        imgTotal = range(trialTotalNum)
        random.shuffle(imgTotal)
        responseTotal = []
        typeOfAnswerTotal = []
        PossibleTreeNumList = [33, 45, 45, 45, 33, 33, 45, 33, 45, 33, 11, 33, 45, 45, 197, 33, 45, 33, 45, 45, 33, 45, 45, 9, 33, 45, 45, 45, 33, 45, 45, 33, 45, 45, 45, 33, 45, 45, 45, 33, 33, 33, 33, 33, 45, 197, 45, 197]

        for imgNum in imgTotal:
            typeOfAnswer = random.randint(1,2)
            typeOfAnswerTotal.append(typeOfAnswer)
            typeOfAnswer = 2
            if typeOfAnswer == 1:
                PathOfAnswer = pathOfExperiment + "human/"
                FilenameOfAnswer = "most_like_tree_" + str(imgNum + 1) + ".gpickle"
            if typeOfAnswer == 2:
                PathOfAnswer = pathOfExperiment + "machine/"
                PossibleTreeNum = PossibleTreeNumList[imgNum]
                FilenamesOfPossibleTrees = ["tree_" + str(imgNum + 1) + '_possible_' + str(TreeIndex) +  ".gpickle" for TreeIndex in range(PossibleTreeNum)]
                PossibleTrees = [networkx.read_gpickle(PathOfAnswer + filenameOfTree) for filenameOfTree in FilenamesOfPossibleTrees]
                
                getCRPCode = getAllTreeCRPCode(all_tree_graph = PossibleTrees)
                AllTreeCRPCode = getCRPCode()
                All_Table_Partion_List = map(lambda x: AllTreeCRPCode[x], range(PossibleTreeNum))
                Prior = prior(all_table_partion_list = All_Table_Partion_List)
                Prior_Probability =  Prior()
                NonLeafNodesInAllTrees = [[ n for n,d in Tree.out_degree().items() if d!=0] for Tree in PossibleTrees]
                CutProportionsInAllTrees = [[Tree.node[NonLeafNode]['cut_proportion'] for NonLeafNode in NonLeafNodesInOneTree] for Tree, NonLeafNodesInOneTree in zip(PossibleTrees, NonLeafNodesInAllTrees)]
                print(CutProportionsInAllTrees[0])
                Likelihood_Probability = [reduce(op.mul, [scipy.stats.dirichlet.pdf(CutProportion, len(CutProportion) * [ALPHA_BASE]) for CutProportion in CutProportionsInOneTree]) for CutProportionsInOneTree in CutProportionsInAllTrees]
                posterior_list = map(lambda x: Prior_Probability[x] * Likelihood_Probability[x], range(PossibleTreeNum))
                posterior_sample_list = map(lambda x: x * 1.0 / numpy.array(posterior_list).sum(), posterior_list)
            
                PossibleTreeIndex = list(numpy.random.multinomial(1, posterior_sample_list)).index(1)
                FilenameOfAnswer = "tree_" + str(imgNum + 1) + '_possible_' + str(PossibleTreeIndex) +  ".gpickle"

            TreeFromAnswer = readTreeFromAnswer(PathOfAnswer, FilenameOfAnswer)
            response = displaySceneParsingImageByTree(TreeFromAnswer, self.lineMaxWidth, self.colorSpace)
            responseTotal.append(response)

        timeNow = time.time()
        numpy.savetxt('data' + str(timeNow) + '.txt',numpy.vstack([range(trialTotalNum), typeOfAnswerTotal, responseTotal]).T, ['%f', '%f','%f'])
        return typeOfAnswerTotal, responseTotal

def main():

    ColorSpace = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255)]
    LineMaxWidth = 16
    ExperimentOfDecision = experimentOfDecision(LineMaxWidth, ColorSpace)

    TrailTotalNum = 48
    PathOfExperiment = "Turing_1_2/"
    ExperimentOfDecision(TrailTotalNum, PathOfExperiment)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
