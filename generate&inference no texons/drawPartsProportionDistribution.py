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
IMG_LIST = np.array([28003])
IMG_NUM_BATCH = 1
IMG_NUM = 10001

"""
ncrp parameters
"""

ALL_GUEST_NUM = 4
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
PROPORTIONLIST = [[proportion, 1-proportion] for proportion in np.arange(0.01, 0.991, 0.01)]
ALPHA = 1
ALPHAList = [0.5, 1, 2]


"""
code parameters
"""

"""
global arguements
"""

def drawPerformanceLine(dataDf, axForDraw, title):
    dataDf.plot(ax=axForDraw, label=title, y='p')


def distributeTree(conditionDf):
    alpha = conditionDf.index.get_level_values('alpha')[0]
    
    proportionLikelihood = [scipy.stats.dirichlet.pdf(cutProportion, [alpha]*len(cutProportion)) for cutProportion in PROPORTIONLIST]

    return pd.DataFrame({'p':proportionLikelihood}, index = pd.Index(list(range(len(PROPORTIONLIST))), name = 'proportion'))

def main(): 
    dfIndex = pd.MultiIndex.from_product([ALPHAList], names = ['alpha'])
    toSplitFrame = pd.DataFrame(index=dfIndex)
    distributionDf = toSplitFrame.groupby(['alpha']).apply(distributeTree)
    
    fig = plt.figure()
    plotRowNum = 1
    plotColNum = 1
    plotCounter = 1

    axForDraw = fig.add_subplot(plotRowNum, plotColNum, plotCounter)
    for (key, dataDf) in distributionDf.groupby('alpha'):
        dataDf.index = dataDf.index.droplevel('alpha')
        drawPerformanceLine(dataDf, axForDraw, str(key))
        plotCounter += 1

    plt.legend(loc='best')

    plt.show()
    
if __name__ == "__main__":
    main()
    

