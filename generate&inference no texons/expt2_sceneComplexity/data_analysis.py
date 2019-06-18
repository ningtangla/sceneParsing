# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 13:06:50 2017

@author: Edward Coen
"""
from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xlrd
import scipy.stats
from scipy.stats import norm as ssn

z_array_all = np.array([0] * 20) 
for i in range(15):
    if i != 2 and i != 9:
        data_all = pd.read_excel('E:/results/' + str(i + 1) + '.xlsx')
        data_key = data_all.loc[:,['img1','img2','pos_ran','Reaction']]
        img1_data = data_key['img1']
        img2_data = data_key['img2']
        select_list = data_key['pos_ran'] - data_key['Reaction']
        select_img_num = [0] * 20           
        for j in range(190):
            if select_list[j] == 1:
                select_img_num[img1_data[j] - 1] = select_img_num[img1_data[j] - 1] + 1
            else:
                select_img_num[img2_data[j] - 1] = select_img_num[img2_data[j] - 1] + 1
        p_value_array = (np.array(select_img_num) + 0.5)/ 20
        z_value_array = map(lambda x: ssn.ppf(x), p_value_array)
        z_array_all = z_array_all + z_value_array
        z = {'z' : z_value_array} 
        print z_value_array
        export_data = pd.DataFrame(z, columns = ['z'])
        export_data.to_csv('E:/results/' + str(i + 1) + '_ana.csv')
    
z_all = {'z_all' : z_array_all} 
export_data_all = pd.DataFrame(z_all, columns = ['z_all'])
export_data_all.to_csv('E:/results/result_ana.csv')