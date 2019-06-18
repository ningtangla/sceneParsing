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

class experimentOfDecision():

    def __init__(self,lineMaxWidth, colorSpace):

        self.lineMaxWidth = lineMaxWidth
        self.colorSpace = colorSpace

    def __call__(self, trialTotalNum, pathOfExperiment):

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

            FilenameOfAnswer = "most_like_tree_" + str(imgNum + 1) + ".gpickle"
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
    PathOfExperiment = "Turing_1_0/"
    ExperimentOfDecision(TrailTotalNum, PathOfExperiment)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
