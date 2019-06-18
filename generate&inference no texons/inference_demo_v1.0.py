# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 13:52:00 2017

@author: Edward Coen
"""

from __future__ import division
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

sys.setrecursionlimit(100000000)
"""
enumerate_tree parameter
"""

TREE_NUM = 3

"""
ncrp parameters
"""

GAMMA = 1
ALL_GUEST_NUM = 6

"""
image parameters
"""

IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 768
COLOR_SPACE = [[128,128,128],[0,0,255],[0,255,0],[255,0,0],[0,255,255],[255,0,255],[255,255,0],[255,255,255]]
GRAPH_WIDTH = 500

"""
Dirchlet parmeters
"""
ALPHA_BASE = [20]


"""
code parameters
"""

"""
global arguements
"""

CODE_BASE = 10    #decimal

class decoder():    
    
    def __init__(self, code_base):
        self.code_base = code_base
        
    def decode_princeple(self, all_guest_num): 
        curr_table_guest_num = self.code_tree[0]
        self.code_tree = self.code_tree[1:]
            
        self.curr_guest_num = self.curr_guest_num + int(curr_table_guest_num)
        self.table_partion_list[len(self.table_partion_list) - 1].append(int(curr_table_guest_num))
        
        if int(curr_table_guest_num) != 1:
            self.all_guest_num_list.append(int(curr_table_guest_num))

        if self.curr_guest_num == all_guest_num:
            self.table_partion_list.append([])
            self.curr_guest_num = 0 
            return
        else:
            return self.decode_princeple(all_guest_num)

    def make_decode_list(self, code_tree):
        self.code_tree = code_tree
        self.table_partion_list = [[]]
        self.all_guest_num_list = [ALL_GUEST_NUM]
        self.curr_guest_num = 0
        map(self.decode_princeple, self.all_guest_num_list)
        del self.table_partion_list[-1]
        self.all_table_partion_list.append(self.table_partion_list)
        
    def __call__(self):
        self.code_tree_list = list(pd.read_csv('E:/ncrp_generate/tree_kind_num_' + str(ALL_GUEST_NUM) + '.csv')['tree_code'])
        self.code_tree_list = map(str, self.code_tree_list)        
        self.all_table_partion_list = []  
        map(self.make_decode_list, self.code_tree_list)
        return self.all_table_partion_list
        
class prior():  
    
    def __init__(self, all_table_partion_list):
        self.all_table_partion_list = all_table_partion_list
    
    def cal_renomalize_parameter(self, table_partion):
        return 1/(1 - 1/np.array(table_partion).sum())
        
    def cal_probability_table_partion(self, table_partion):
        return reduce(op.mul, map(math.factorial, (np.array(table_partion) - 1)))/math.factorial(np.array(table_partion).sum())
            
    def cal_permutation_table_partion(self, table_partion):
        return list(set(list(itertools.permutations(table_partion))))
    
    def cal_all_combination_guest(self, permutation_table_partion): 
        return reduce(op.add, map(self.cal_permutation_combination_guest, permutation_table_partion))
        
    def cal_permutation_combination_guest(self, table_partion):
        self.guest_left = np.array(table_partion).sum()
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
#        print reduce(op.mul, np.array(probability_table_partion)*np.array(all_combination_guest)*np.array(renomalize_parameter))
        return reduce(op.mul, np.array(probability_table_partion)*np.array(all_combination_guest)*np.array(renomalize_parameter))
        
    def __call__(self):
        return map(self.cal_prior_probability, self.all_table_partion_list)
        
class likelihood():
    def __init__(self, all_table_partion_list, color_space, alpha_base):
        self.all_table_partion_list = all_table_partion_list
        self.alpha_base = alpha_base
        self.color_space = color_space
    
    def find_all_vertex(self, color_space):
        self.all_vertex_list = []
        map(self.find_vertex, color_space)
        
    def find_vertex(self, color):
        lower = np.array(color, dtype = "uint8")
        upper = np.array(color, dtype = "uint8")
        mask = cv2.inRange(self.img, lower, upper)
        index = np.argwhere(mask == 255)
        if len(index) != 0:
            self.all_vertex_list.extend([index.min(axis = 0), index.max(axis = 0)])
            
    def detect_cut_point_list(self, vertex):
        min_x, min_y = np.array(vertex).min(axis = 0)
        max_x, max_y = np.array(vertex).max(axis = 0) 
        all_vertex_array = np.array(self.all_vertex_list)
        x_vertex_array = all_vertex_array[:, 1]
        y_vertex_array = all_vertex_array[:, 0]
        inner_vertex_array = all_vertex_array[(np.where((x_vertex_array >= min_x) & (x_vertex_array <= max_x))) or (np.where((y_vertex_array >= min_y) & (y_vertex_array <= max_y)))]
        inner_vertex_list = map(tuple, inner_vertex_array)
        inner_vertex_array = np.array(list(set(inner_vertex_list).difference(set([(min_y, min_x), (max_y, max_x)]))))
        
        if len(inner_vertex_array) == 0:
            vertx_array_y = []
            vertx_array_x = []
        else:
            inner_x_vertex_array = inner_vertex_array[:, 1]
            inner_y_vertex_array = inner_vertex_array[:, 0]
#            print  inner_x_vertex_array, inner_y_vertex_array 
            
            x_min_vertex_array_y = inner_vertex_array[np.where(np.abs(inner_x_vertex_array - min_x) < 3)][:, 0] 
            x_max_vertex_array_y = inner_vertex_array[np.where(np.abs(inner_x_vertex_array - max_x) < 3)][:, 0] + 1
            y_min_vertex_array_x = inner_vertex_array[np.where(np.abs(inner_y_vertex_array - min_y) < 3)][:, 1]
            y_max_vertex_array_x = inner_vertex_array[np.where(np.abs(inner_y_vertex_array - max_y) < 3)][:, 1] + 1
#            print '&&&'
#            print x_min_vertex_array_y, x_max_vertex_array_y, y_min_vertex_array_x, y_max_vertex_array_x, 
            vertx_array_y =  np.intersect1d(x_min_vertex_array_y, x_max_vertex_array_y)
            vertx_array_x =  np.intersect1d(y_min_vertex_array_x, y_max_vertex_array_x)
#            print(vertx_array_y)
#            print(vertx_array_x)
            
        if (len(vertx_array_y) !=0) or (len(vertx_array_x) !=0):
            if len(vertx_array_y) == 0 :
                min_vertex = min_x
                max_vertex = max_x
                cut_point_list = list(vertx_array_x)
                cut_point_list.sort()
#                print '!!!'
#                print cut_point_list
                new_vertex_x = map(lambda x: [cut_point_list[x], cut_point_list[x + 1] - 1], list(range(len(cut_point_list) - 1)))
                new_vertex_x.insert(0, [min_x, cut_point_list[0] - 1])
                new_vertex_x.append([cut_point_list[-1], max_x])
                new_vertex_y = [[min_y, max_y]] * len(new_vertex_x)
                
            else:
                min_vertex = min_y
                max_vertex = max_y
                cut_point_list = list(vertx_array_y)
#                print '!!!'
#                print cut_point_list
                cut_point_list.sort()
                new_vertex_y = map(lambda x: [cut_point_list[x], cut_point_list[x + 1] - 1], list(range(len(cut_point_list) - 1)))
                new_vertex_y.insert(0, [min_y, cut_point_list[0] -1])
                new_vertex_y.append([cut_point_list[-1], max_y])
                new_vertex_x = [[min_x, max_x]] * len(new_vertex_y)
                
            
            new_vertex_list = map(zip, new_vertex_x, new_vertex_y)
            propotion_list = list((np.array(cut_point_list)-min_vertex)/((max_vertex - min_vertex)*1.0)) 
            
            cut_propotion_list = map(lambda x: propotion_list[x+1] - propotion_list[x], range(len(propotion_list) - 1))
            cut_propotion_list.insert(0, propotion_list[0] - 0)
            cut_propotion_list.append(1 - propotion_list[-1])
            
        else:
            cut_point_list = []
            cut_propotion_list = []
            new_vertex_list = []
        
        return cut_point_list, cut_propotion_list, new_vertex_list
        
    def cal_p_dirichlet(self, cut_propotion):
        alpha = self.alpha_base * len(cut_propotion)
        return scipy.stats.dirichlet.pdf(cut_propotion, alpha)
    
    def cal_p_table_partion(self, table_partion):
        self.x = 1
        self.cut = np.array(map(self.detect_cut_point_list, self.vertex))
        self.cut_point_list = list(self.cut[:, 0])[0]
        self.cut_propotion_list = list(self.cut[:, 1])[0]
        self.table_partion = table_partion
        if table_partion.count(1) != len(table_partion):
            self.next_level_table_partion = map(lambda x: self.table_partion_list[x], np.array(range(len(table_partion) - table_partion.count(1))) + self.x + 1)
            self.x = self.x + len(table_partion) - table_partion.count(1)
            print '###'
            print self.next_level_table_partion
    #        print self.cut_propotion_list
            self.new_vertex_list = list(self.cut[:, 2])[0]
            print self.new_vertex_list
            self.permutation_table_partion_index = list(itertools.permutations(range(len(table_partion))))
            
    #                                                              
    #        self.next_level_table_partion = map(self.table_partion_list[self.x + np.array(range(len(table_partion_filtered)))]
    #        
            self.permutation_table_partion = map(self.cal_permutation_table_partion, self.permutation_table_partion_index)
            print self.permutation_table_partion_index
            self.p_list = map(self.cal_cut_point_and_corresponding_table_partion, self.permutation_table_partion)
#            self.z = map(lambda x: map(list, x), self.p_list)
            print self.p_list
            self.ttt = np.array(self.p_list)
            print '!!!', np.shape(self.ttt[0][0])
            self.y = map(self.get_part, range(len(self.p_list)), [0]*len(self.p_list))
            self.yy = map(self.get_part, range(len(self.p_list)), [1]*len(self.p_list))
            if len(self.yy[0][0]) != 0:
                print self.permutation_table_partion_index
                self.find_next_vertex_index = map(lambda x: map(lambda y: x.index(y), np.array(range(len(self.next_level_table_partion))) + table_partion.count(1)), self.permutation_table_partion_index)
                print self.find_next_vertex_index
                self.next_table_partion_list = map(lambda x: map(lambda y: self.next_level_table_partion[(np.array(x).argsort())[y]], range(len(self.next_level_table_partion))), self.find_next_vertex_index)
                print 'tttt',self.next_table_partion_list
                self.next_vertex = map(lambda x: map(lambda y: map(lambda z: self.yy[x][y][z], self.find_next_vertex_index[x]), range(len(self.yy[0]))), range(len(self.yy)))
                self.next_permutation = map(lambda x: map(list,(map(itertools.permutations, x))), self.next_table_partion_list)
                print self.next_permutation
                print '000', self.next_table_partion_list
#                print self.next_permutation
                self.next_permutation2 = map(lambda x: list(itertools.product(*x)), self.next_permutation)
                print self.next_permutation2
                self.next_cut = map(lambda x: map(lambda y: map(self.detect_cut_point_list, self.next_vertex[x][y]), range(len(self.next_vertex[0]))), range(len(self.next_vertex))), 
                print list(np.array(self.next_cut)[:,:,:,:,0])[0]
                self.next_cut_point_list = list(np.array(self.next_cut)[:,:,:,:,0])[0] 
                self.next_cut_propotion = list(np.array(self.next_cut)[:,:,:,:,1])[0]
                self.next_new_vertex_list = list(np.array(self.next_cut)[:,:,:,:,2])[0]
                print 'zzz', len(self.next_cut_point_list[0]), self.next_cut_point_list[0]
#                print self.next_cut_point_list, self.next_cut_propotion
                self.next_p_list = map(lambda (x, y, z): self.cal_cut_point_and_corresponding_table_partion2(x, y, z), list(itertools.product(range(len(self.next_permutation2)), range(len(self.next_cut_point_list[0])), range(len(self.next_permutation2[0])))))
                self.next_ttt = np.array(self.next_p_list)
#                print '!!!', self.next_ttt[:,:,0]
                print self.next_p_list
#                self.z = map(lambda x: map(lambda y: map(list, x[y]), range(len(self.next_p_list[0]))), self.next_p_list)
                

#                self.y = map(self.get_part, range(len(self.z)), [0]*len(self.z))
#                self.yy = map(self.get_part, range(len(self.z)), [1]*len(self.z))
#                print self.yy
#                self.pppppp = 
    #        self.now_table_partion_index = 1
    #        self.next_level_tabel_partion = map(self.cal_next_level_table_partion, np.array(self.combination_index_list) + self.now_table_partion_index )
    #        self.undividable_table_partion = self.table_partion.count(1) - 1
    #        self.all_p_list.append(self.p_list)
#            print self.p_list
    #        p_filtered = map(self.filter_p_positive, self.p_list)
    #        print p_filtered
    #        self.p_value_num = map(len, p_filtered)
    #        self.p_value_positive_index = list(np.where(np.array(self.p_value_num) > 0)[0])
    #        print self.p_value_num, self.p_value_positive_index
    #        print self.permutation_table_partion
            return self.p_list
    #    def get_next_vertex(self, index):
    
#    def get_permutation_flatten(self, permutation_list):        
#        if type(permutation_list) != int:
#            self.d.insert(0, permutation_list[-1])
#            permutation_list = permutation_list[0]
#        else:
#            temp = self.d
#            
#            self.d = []
#            return temp
        
    def get_part(self, target_from_index, target_at_index):
        target_list = self.p_list[target_from_index]
        return map(lambda c: target_list[c][target_at_index], range(len(target_list)))
        
    def filter_p_positive(self, target):
        return filter(lambda x: x>0, target)
    
    def cal_cut_point_and_corresponding_table_partion2(self, x, y, z):
        print x,y,z
        table_partion = list(self.next_permutation2[x][z])
        cut_point_list = self.next_cut_point_list[x][y]
        cut_propotion = self.next_cut_propotion[x][y]
        new_vertex_list = self.next_new_vertex_list[x][y]
        print table_partion, cut_point_list, cut_propotion, new_vertex_list
        self.t = map(lambda x: self.cal_cut_point_and_corresponding_table_partion3(table_partion[x], cut_point_list[x], cut_propotion[x], new_vertex_list[x]), range(len(table_partion)))
#        self.p = map(lambda x, y )
        tt = map(lambda x: self.flatten(x), range(len(self.t)))

        return tt
   
    def flatten(self, t_index):
        if t_index == 0:
            self.p = map(lambda x: self.t[t_index][x][0], range(len(self.t[t_index])))
            
            self.ne_vertex = map(lambda x: self.t[t_index][x][1], range(len(self.t[t_index])))
        else:
            self.pp = [[0]] * len(self.p) * len(self.t[t_index])
            self.nne_vertex = [[0]] * len(self.p) * len(self.t[t_index])
            map(lambda (x, y) : self.new_assign_value(x, y, t_index), list(itertools.product(range(len(self.t[t_index])), range(len(self.p)))))
#            self.ne_vertex = map(lambda x: map(lambda y: self.ne_vertex[y].append(self.t[t_index][x][1]), range(len(self.ne_vertex))), range(len(self.t[t_index])))
            self.p = self.pp
            self.ne_vertex = self.nne_vertex
        return self.p, self.ne_vertex
   
    def new_assign_value(self, x, y, t_index):
        self.pp[x * len(self.p) + y] = [self.p[y][0] * self.t[t_index][x][0][0]]
        if self.pp[x * len(self.p) + y][0] == 0:
            self.nne_vertex[x * len(self.p) + y] = []
        else:
            self.nne_vertex[x * len(self.p) + y] = [self.ne_vertex[y], self.t[t_index][x][1]]
        print self.pp, self.nne_vertex
        
    def cal_cut_point_and_corresponding_table_partion3(self, table_partion, cut_point_list, cut_propotion, new_vertex_list):
        self.table_partion = table_partion
        self.cut_point_list = cut_point_list
        self.cut_propotion = cut_propotion
        self.new_vertex_list = new_vertex_list
        if (len(self.table_partion) - 1) > len(self.cut_point_list):
            self.result = [[[0], [], []]]
            
        else:
            self.combination_index = list(itertools.combinations(range(len(self.cut_point_list)), (len(self.table_partion) - 1)))
#            print self.cut_point_list, self.new_vertex_list
#            print table_partion
#            print self.combination_index
#            self.combination_index_list = [self.combination_index] * len(self.permutation_table_partion)
            
#            print self.combination_index_list
            self.result = map(self.cal_p_cut_propotion, self.combination_index)
        return self.result
        
    def cal_cut_point_and_corresponding_table_partion(self, table_partion):
        if (len(table_partion) - 1) > len(self.cut_point_list):
            self.result = [[[0], [], []]]
            
        else:
            self.combination_index = list(itertools.combinations(range(len(self.cut_point_list)), (len(self.table_partion) - 1)))
#            print self.cut_point_list, self.new_vertex_list
#            print table_partion
#            print self.combination_index
#            self.combination_index_list = [self.combination_index] * len(self.permutation_table_partion)
            
#            print self.combination_index_list

            self.result = map(self.cal_p_cut_propotion, self.combination_index)
        return self.result
        
#    def cal_next_level_table_partion(self, table_partion_index)
        
    def cal_p_cut_propotion(self, index):
#        cut_point = map(lambda x:self.cut_point_list[x], list(index))
#        print index
        new_vertex = self.cal_new_vertex(index)
#        print new_vertex
        cut_propotion = self.cal_cut_propotion(index)
#        print cut_propotion
        next_cut = np.array(map(self.detect_cut_point_list, new_vertex))
        next_cut_point_num = map(len, list(next_cut[:, 0]))
#        print self.table_partion, next_cut_point_num
        diff = map(lambda(x, y): x - y, zip(self.table_partion, next_cut_point_num))
#        print diff
        if len(filter(lambda x: x <= 0, diff)) > 0:
            return [[0], new_vertex, list(index)]
        else:
            return [[self.cal_p_dirichlet(cut_propotion)], new_vertex, list(index)]
                
                               
#    def cal_combination_cut_point(self, cut_point_index):
#        return map(lambda x:self.cut_point_list[x], cut_point_index)
#    
    def cal_permutation_table_partion(self, table_partion_index):
        return map(lambda x:self.table_partion[x], table_partion_index)
    
    def cal_cut_propotion(self, propotion_index):
        if len(propotion_index) == (len(self.cut_propotion_list) - 1):
            cut_propotion = self.cut_propotion_list
        else:
            cut_propotion = map(lambda x: np.array(self.cut_propotion_list)[propotion_index[x] + 1:propotion_index[x + 1] + 1].sum(), range(len(propotion_index) - 1))
            cut_propotion.insert(0, np.array(self.cut_propotion_list)[0:propotion_index[0] + 1].sum())
            cut_propotion.append(1 - np.array(self.cut_propotion_list)[0:propotion_index[-1] + 1].sum())
        return cut_propotion
            
    def cal_new_vertex(self, vertex_index):
        if len(vertex_index) == (len(self.new_vertex_list) - 1):
            new_vertex = self.new_vertex_list
        else:
            new_vertex = map(lambda x: [self.new_vertex_list[vertex_index[x] + 1][0], self.new_vertex_list[vertex_index[x + 1]][1]], range(len(vertex_index) - 1))
            new_vertex.insert(0, [self.new_vertex_list[0][0], self.new_vertex_list[vertex_index[0]][1]])
            new_vertex.append([self.new_vertex_list[vertex_index[-1] + 1][0], self.new_vertex_list[-1][1]])
        return new_vertex
    
    def cal_p_one_table_partion_list(self, table_partion_list):
        self.c = self.c + 1
        print '***'
        print self.c
        self.table_partion_list = table_partion_list
        self.vertex = [[(0, 0), (1023, 767)]]
        self.all_vertex_list = []
        map(self.find_vertex, self.color_space)
        p = self.cal_p_table_partion(self.table_partion_list[1])
#        print p
        return p
    
    def cal_likelihood(self):
        self.c = 0 
        map(self.cal_p_one_table_partion_list, self.all_table_partion_list[4:6])
        
    def __call__(self, img_num):
        self.img = cv2.imread('E:/ncrp/'+str(img_num)+'.png')
        self.cal_likelihood()
        if img_num == 3:
            return
        else:
            return self.__call__(img_num - 1)
        
        
def main():           
    Decode_list = decoder(code_base = CODE_BASE)
    All_Table_Partion_List = Decode_list()
    Prior = prior(All_Table_Partion_List)
    Prior_Probability =  Prior()
    Likelihood = likelihood(All_Table_Partion_List, COLOR_SPACE, ALPHA_BASE)
    Likelihood_Probability = Likelihood(img_num = TREE_NUM )
    
    
if __name__ == "__main__":
    main()
    

