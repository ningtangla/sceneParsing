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

def readTreeFromAnswer(pathOfAnswer, filenameOfAnswer):
    treeFromAnswer = networkx.read_gpickle(pathOfAnswer + filenameOfAnswer)
    return treeFromAnswer
   
def displaySceneParsingImageByTree(tree, lineMaxWidth, colorSpace):
    nonleafNodes = [nodeID for nodeID, nodeOutDegree in tree.out_degree().items() if nodeOutDegree != 0]

    imageWidth = numpy.max(tree.node[1]['x'])
    imageHeight = numpy.max(tree.node[1]['y'])
    image = numpy.zeros([imageHeight, imageWidth, 3], 'uint8') 
    cv2.namedWindow('image')
    cv2.imshow('image', image)
    cv2.waitKey()
    
    depthOfLastNode = 0
    for currNode in nonleafNodes:
        depthOfCurrentNode = tree.node[currNode]['depth']
        childrenOfCurrNode = tree.successors(currNode)
        childrenOfCurrNode.sort()
        
        if tree.node[currNode]['direction'] == 0:
            linesCoordinates = map(lambda z: [(tree.node[z]['x'][1] , tree.node[currNode]['y'][0]), 
                                              (tree.node[z]['x'][1] , tree.node[currNode]['y'][1])],
                                             childrenOfCurrNode[ :-1])
        if tree.node[currNode]['direction'] == 1:
            linesCoordinates = map(lambda z: [(tree.node[currNode]['x'][0] , tree.node[z]['y'][1]), 
                                              (tree.node[currNode]['x'][1] , tree.node[z]['y'][1])],
                                             childrenOfCurrNode[ :-1])
        for lineCoordinate in linesCoordinates:
            cv2.line(image, lineCoordinate[0], lineCoordinate[1], colorSpace[tree.node[currNode]['depth'] - 1], int(lineMaxWidth * numpy.power(0.5, tree.node[currNode]['depth'])))    
        cv2.imshow('image', image)
            
        if depthOfCurrentNode > depthOfLastNode:
            response = cv2.waitKey()
            
        depthOfLastNode = depthOfCurrentNode
        return response
    
class experimentOfDecision():
    
    def __init__(self,colorSpace, lineMaxWidth):
        self.colorSpace = colorSpace
        self.lineMaxWidth = lineMaxWidth
        
    def __call__(self,trialTotalNum, pathOfExperiment):
        
        imgTotal = range(trialTotalNum)
        random.shuffle(imgTotal)
        responseTotal = []
        typeOfAnswerTotal = []
        
        for imgNum in imgTotal:
            typeOfAnswer = random.randint(1,2)
            typeOfAnswerTotal.append(typeOfAnswer)
            if typeOfAnswer == 1:
                PathOfAnswer = pathOfExperiment + "human/"
            if typeOfAnswer == 2:
                PathOfAnswer = pathOfExperiment + "machine/"
            
    #    PathOfAnswer = "test/"
            FilenameOfAnswer = str(imgNum) + ".gpickle"
            TreeFromHumanAnswer = readTreeFromAnswer(PathOfAnswer, FilenameOfAnswer)
            response = displaySceneParsingImageByTree(TreeFromHumanAnswer, self.lineMaxWidth, self.colorSpace)
            responseTotal.append(response)
            
        numpy.savetxt('data.txt',numpy.vstack([typeOfAnswerTotal, responseTotal]).T,['%f','%f'])
        return typeOfAnswerTotal, responseTotal

def main():    
    
    ColorSpace = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255)]
    LineMaxWidth = 32
    ExperimentOfDecision = experimentOfDecision(LineMaxWidth, ColorSpace)
    
    TrailTotalNum = 48
    pathOfExperiment = "Turing_2_0/"
    ExperimentOfDecision(TreeFromHumanAnswer)
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()