# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 20:43:20 2017

@author: Edward Coen
"""
import numpy as np
import cv2 
import pandas as pd

COLOR_SPACE = [[128,128,128],[0,0,255],[0,255,0],[255,0,0],[0,255,255],[255,0,255],[255,255,0],[255,255,255]]
IMAGE_SPACE = [10009,10021,10032,10050,10086,10100,10113,10119,10143,10154,10185,10224,10381,10412,10445,10576,10624,10640,10818,10835]
variance_list = []
std_list = []

for img_num in IMAGE_SPACE:
    img = cv2.imread('E:/ncrp_generate/'+str(img_num)+'.png')
    size_list = []
    print '111'
    for color in COLOR_SPACE:
        lower = np.array(color, dtype = "uint8")
        upper = np.array(color, dtype = "uint8")
        mask = cv2.inRange(img, lower, upper)
        index = np.argwhere(mask == 255)
        if len(index) != 0:
            size_list.append(len(index))
        else:
            print color
    
    size_array = np.array(size_list)
    variance_list.append(np.var(size_array))
    std_list.append(np.std(size_array))

size_var_std = {'var': variance_list, 'std': std_list}    
export_var_std = pd.DataFrame(size_var_std, columns = ['var', 'std'])
export_var_std.to_csv('E:/pic_IC_and_CE/size_var_std.csv')
print '!!!', variance_list, std_list