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

subjectIndex = int(input("subjectIndex: "))
imgList = numpy.array([95,409,545,634,695,991,1260,2067,2116,2240,2541,2656,2788,3007,3354,3395,3596,3799,4119,4510]) + 10000

def displaySceneParsingImageByTree(pathOfAnswer, filenameOfAnswer, lineMaxWidth, colorSpace):
    tree = networkx.read_gpickle(pathOfAnswer + filenameOfAnswer+'.gpickle')
    nonleafNodes = [nodeID for nodeID, nodeOutDegree in dict(tree.out_degree()).items() if nodeOutDegree != 0]
    imageWidth = 0.8 * numpy.max(tree.node[1]['x'])
    imageHeight = numpy.max(tree.node[1]['y'])
    image = numpy.zeros([int(imageHeight), int(imageWidth), 3], 'uint8')
    image[ int(0.8*imageHeight) : int(imageHeight), : , :] = 255
    cv2.namedWindow('image')
    cv2.imshow('image', image)

#    depthOfLastNode = 1
#    for currNode in nonleafNodes:
#        depthOfCurrentNode = tree.node[currNode]['depth']
#        childrenOfCurrNode = list(tree.successors(currNode))
#        childrenOfCurrNode.sort()
#
#        if tree.node[currNode]['direction'] == 0:
#            linesCoordinates = map(lambda z: [(tree.node[z]['x'][1], tree.node[currNode]['y'][0]),
#                                              (tree.node[z]['x'][1], tree.node[currNode]['y'][1])],
#                                             childrenOfCurrNode[ :-1])
#        if tree.node[currNode]['direction'] == 1:
#            linesCoordinates = map(lambda z: [(tree.node[currNode]['x'][0], tree.node[z]['y'][1]),
#                                              (tree.node[currNode]['x'][1], tree.node[z]['y'][1])],
#                                             childrenOfCurrNode[ :-1])
#
#        if depthOfCurrentNode > depthOfLastNode:
#            cv2.waitKey()
#            cv2.imshow('image', image)
#
#        for lineCoordinate in linesCoordinates:
#            cv2.line(image, (int(0.8 * lineCoordinate[0][0]), int(0.8 * lineCoordinate[0][1])), (int(0.8 * lineCoordinate[1][0]), int(0.8 * lineCoordinate[1][1])), colorSpace[tree.node[currNode]['depth'] - 1], max(2, int(lineMaxWidth * numpy.power(0.5, tree.node[currNode]['depth']))))
#
#        depthOfLastNode = depthOfCurrentNode
#
#    cv2.waitKey()
#    cv2.imshow('image', image)

    sceneAndTreeImageFromAnswer = cv2.imread(pathOfAnswer + filenameOfAnswer+'.png')
    scaledSceneAndTreeImage = cv2.resize(sceneAndTreeImageFromAnswer, (int(1*imageWidth), int(0.805*imageHeight)), interpolation = cv2.INTER_AREA)
    image[:int(0.805*imageHeight), : , :] = scaledSceneAndTreeImage
    questionImg = cv2.imread('question.png')
    image[int(0.9*imageHeight) - 69: int(0.9*imageHeight) + 77, int(0.5*imageWidth) - 242: int(0.5*imageWidth) + 242] = questionImg
    cv2.waitKey()
    cv2.imshow('image', image)
    
    response = None
    while (response != ord('f')) and (response != ord('j')):
        response = cv2.waitKey()
    image = numpy.zeros([int(imageHeight), int(imageWidth), 3], 'uint8')
    cv2.imshow('image', image)
    typeOfResponse = 0
    if response == ord('f'):
        typeOfResponse = 1 
    if response == ord('j'):
        typeOfResponse = 2 
    return typeOfResponse 

class experimentOfDecision():

    def __init__(self,lineMaxWidth, colorSpace):

        self.lineMaxWidth = lineMaxWidth
        self.colorSpace = colorSpace

    def __call__(self, trialTotalNum, pathOfExperiment):

        imgTotal = range(trialTotalNum)
        random.shuffle(imgTotal)
        imageIndexTotal = []
        responseTotal = []
        typeOfAnswerTotal = []

        for imgNum in imgTotal:
            typeOfAnswer = random.randint(1,2)
            typeOfAnswerTotal.append(int(typeOfAnswer))
            if typeOfAnswer == 1:
                PathOfAnswer = pathOfExperiment + "humanAnswer/"
            if typeOfAnswer == 2:
                PathOfAnswer = pathOfExperiment + "nonformativePrior/"

            FilenameOfAnswer = str(imgList[imgNum]) +'sub' + str(subjectIndex)
            response = displaySceneParsingImageByTree(PathOfAnswer, FilenameOfAnswer, self.lineMaxWidth, self.colorSpace)
            responseTotal.append(int(response))
            imageIndexTotal.append(int(imgList[imgNum]))

        timeNow = time.time()
        numpy.savetxt('data/Turing2-1/sub' + str(subjectIndex) + '.txt',numpy.vstack([imageIndexTotal, typeOfAnswerTotal, responseTotal]).T, ['%f', '%f','%f'])
        return typeOfAnswerTotal, responseTotal

def main():

    ColorSpace = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255)]
    LineMaxWidth = 16
    ExperimentOfDecision = experimentOfDecision(LineMaxWidth, ColorSpace)

    TrailTotalNum = 20
    PathOfExperiment = ""
    ExperimentOfDecision(TrailTotalNum, PathOfExperiment)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
