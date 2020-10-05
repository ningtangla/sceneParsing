# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 14:09:00 2017

@author: Edward Coen
"""

from __future__ import division
import scipy.stats
import numpy as np
import networkx as nx
from networkx.algorithms.traversal.depth_first_search import dfs_successors
import itertools as it
import operator as op
import pandas as pd
import cv2 
import sys   
import random

sys.setrecursionlimit(100000000)
"""
enumerate_tree parameter
"""

TREE_NUM = 100
IMG_NUM_STRAT = 28001

"""
ncrp parameters
"""
MIN_GAMMA = 1
MAX_GAMMA = 1
GAMMA = [MIN_GAMMA, MAX_GAMMA]
ALL_GUEST_NUM = 4
GAMMA_RECORD = GAMMA
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
MIN_ALPHA_BASE = 10
MAX_ALPHA_BASE = 10
ALPHA_BASE = [MIN_ALPHA_BASE, MAX_ALPHA_BASE]
ALPHA_RECORD = ALPHA_BASE
"""
code parameters
"""

CODE_BASE = 10    #decimal

class Enumerate_tree():
    def __init__(self, gamma, all_guest_num):                    
        self.all_guest_num = all_guest_num             
        self.max_graph_depth = all_guest_num
        self.gamma_range = gamma   
        self.all_tree_graph = []
        
    def creat_new_tree(self):
        self.graph = nx.DiGraph()
        self.graph.add_node(1, guest = range(self.all_guest_num), depth = [1])
        self.start_node = 1
        self.gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
        GAMMA_RECORD.append(self.gamma)
        self.to_process_node = [self.start_node]
        
    def nested_chinese_resturant(self):
        self.creat_new_tree()
        map(self.chinese_resturant, self.to_process_node)
        self.all_tree_graph.append(self.graph)
        
    def chinese_resturant(self, curr_node):        
        self.process_node = curr_node
        self.process_guest  = self.graph.node[curr_node]['guest']
        if len(self.process_guest ) != 1 :
            map(self.assign_table, self.process_guest)
        
    def assign_table(self, curr_guest):
        self.total_guest_num = self.process_guest.index(curr_guest) + 1    
        children = list(nx.DiGraph.successors(self.graph, self.process_node))
        if self.graph.node[self.process_node]['depth'][0] != self.max_graph_depth :
            
            """           
            creat 1st table if no table exists
            cal the probablity of the exist table
            cal the probablity of assign a new table 
            """
            
            if not children:
                self.graph.add_node(len(self.graph.nodes())+1, guest = [], depth = [self.graph.node[self.process_node]['depth'][0]+1])
                self.graph.add_edge(self.process_node, len(self.graph.nodes())) 
                self.to_process_node.append(len(self.graph.nodes()))
                self.p_table = []
                children.append(len(self.graph.nodes()))
            else:
                self.p_table = map(self.cal_p_table, children)
            self.p_table.append(self.gamma/(self.total_guest_num - 1 + self.gamma))    
            
            """
            assign the guest to table by sample from multinomial distribution
            """
            
            table_id = list(np.random.multinomial(1, self.p_table)).index(1)
            if table_id == len(children) :
                self.graph.add_node(len(self.graph.nodes())+1, guest = [], depth = [self.graph.node[self.process_node]['depth'][0]+1])
                self.graph.add_edge(self.process_node, len(self.graph.nodes()))
                self.to_process_node.append(len(self.graph.nodes()))
                children.append(len(self.graph.nodes()))
            self.graph.node[children[table_id]]['guest'].append(curr_guest)        
            
            if (curr_guest == self.process_guest[-1]) and (len(children) == 1):
                self.to_process_node.remove(len(self.graph.nodes()))
                self.graph.remove_edge(self.process_node, len(self.graph.nodes()))
                self.graph.remove_node(len(self.graph.nodes()))
                map(self.assign_table, self.process_guest)
                
                
    def cal_p_table(self, table):
        self.table_guest_num = len(self.graph.node[table]['guest'])
        return self.table_guest_num/(self.total_guest_num - 1 + self.gamma)
        
    def __call__(self, tree_num):
        self.nested_chinese_resturant()
        if tree_num == 1:
            return self.all_tree_graph
        else:
            return self.__call__(tree_num - 1)
        
class Generate_image_by_tree():    
    def __init__(self, image_width, image_height, alpha_base, color_space, graph_width):
        self.graph_width = graph_width
        self.max_graph_depth = ALL_GUEST_NUM
        self.image_width = image_width
        self.image_height = image_height
        self.color_space = color_space
        self.alpha_base_range = alpha_base
        self.img_num = IMG_NUM_STRAT
        self.img_num_nocolor = IMG_NUM_STRAT
        self.node_radius = 15
        self.graph_start_x = graph_width/2
        self.graph_start_y = image_height/(ALL_GUEST_NUM + 1)

        
    def nested_parse_vertex(self, tree_graph) :
        self.img = np.zeros([self.image_height, self.image_width + self.graph_width, 3], 'uint8')
        self.img_nocolor = np.zeros([self.image_height, self.image_width, 3], 'uint8')
        self.alpha = random.uniform(self.alpha_base_range[0], self.alpha_base_range[1])
        ALPHA_RECORD.append(self.alpha)
        self.alpha_base = [self.alpha]
        self.graph = tree_graph
        self.start_node = 1
        self.to_process_node = [self.start_node]
        np.random.shuffle(self.color_space)
        self.color_selection = 0
        self.vertex_list = [[], [(0.0, 0.0), (self.image_width * 1.0, self.image_height * 1.0)]]
#        cv2.rectangle(self.img, 
#                      (int(self.vertex_list[self.start_node][0][0]), int(self.vertex_list[self.start_node][0][1])), 
#                      (int(self.vertex_list[self.start_node][1][0]), int(self.vertex_list[self.start_node][1][1])),
#                      (0, 255, 0), 3)
        
        self.intervel_x_list = [0, self.graph_width]
        self.intervel_y = self.image_height/(ALL_GUEST_NUM + 1)
        self.node_draw_x_list = [self.graph_start_x, self.graph_start_x]
        self.node_draw_y_list = [self.graph_start_y, self.graph_start_y]
        self.font_parm = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
#        cv2.circle(self.img, (int(self.node_draw_x_list[1] + self.image_width), int(self.node_draw_y_list[1])), self.node_radius, (255, 255, 255), -1)
#        cv2.putText(self.img, 
#                    str(len(self.graph.node[self.start_node]['guest'])), 
#                    (int(self.node_draw_x_list[self.start_node] + self.image_width - self.node_radius/2), int(self.node_draw_y_list[self.start_node] + self.node_radius/2)), 
#                    self.font_parm, 0.8, (0, 0, 255), 1)
#        cv2.imshow('demo', self.img)
#        cv2.waitKey() 
        
        map(self.parse_vertex_and_draw, self.to_process_node)
#        cv2.imshow('demo', self.img)
#        cv2.waitKey()
        self.img_save = self.img[0:self.image_height, 0:self.image_width]
        cv2.imwrite('demo_color/'+str(self.img_num)+'.png', self.img_save)
        cv2.imwrite('demo_nocolor/'+str(self.img_num_nocolor)+'.png', self.img_nocolor)
        self.img_num = self.img_num + 1
        self.img_num_nocolor = self.img_num
        
    def parse_vertex_and_draw(self, process_node):
        self.propotion = 0
        self.propotion_tuple = [self.propotion]
        self.propotion_list = []
        self.children = list(nx.DiGraph.successors(self.graph, process_node))
        self.children.sort()
        
#        cv2.rectangle(self.img,
#                      (int(self.node_draw_x_list[process_node - 1] + self.image_width - self.node_radius), int(self.node_draw_y_list[process_node - 1] - self.node_radius)),
#                      (int(self.node_draw_x_list[process_node - 1] + self.image_width + self.node_radius), int(self.node_draw_y_list[process_node - 1] + self.node_radius)),
#                      (255, 0, 0), 2)
#        cv2.rectangle(self.img,
#                      (int(self.node_draw_x_list[process_node] + self.image_width - self.node_radius), int(self.node_draw_y_list[process_node] - self.node_radius)),
#                      (int(self.node_draw_x_list[process_node] + self.image_width + self.node_radius), int(self.node_draw_y_list[process_node] + self.node_radius)),
#                      (0, 255, 0), 2)
        
        if self.children:
                        
            self.intervel_x = self.intervel_x_list[process_node]
            self.intervel_x_list.extend([self.intervel_x/len(self.children)] * len(self.children))
            self.center_node_x = self.node_draw_x_list[process_node]
            self.left_intervel_x = self.center_node_x - self.intervel_x/2            
            
            self.to_process_node.extend(self.children) 
            cut_propotion = self.cal_cut_propotion(self.children)[0]
            map(self.make_propotion, cut_propotion)
            self.cut_direction = np.random.randint(0, 2)   # 0:virtual; 1:horizen
            min_x, min_y = np.array(self.vertex_list[process_node]).min(axis = 0)
            max_x, max_y = np.array(self.vertex_list[process_node]).max(axis = 0)
            x_length = max_x - min_x
            y_length = max_y - min_y
            if self.cut_direction == 0:
                vertex_x = np.array(self.propotion_list)*x_length + min_x
                vertex_y = np.array([[min_y, max_y]]*len(self.propotion_list))
            else:
                vertex_y = np.array(self.propotion_list)*y_length + min_y
                vertex_x = np.array([[min_x, max_x]]*len(self.propotion_list))
            self.vertex = map(zip, vertex_x, vertex_y)
            np.random.shuffle(self.vertex)
            self.vertex_list.extend(self.vertex)
            
            map(self.child_node_rectangle_draw, self.children)
#            cv2.imshow('demo', self.img)
#            cv2.waitKey()
            
        if (not self.children) or (self.graph.node[process_node]['depth'] == self.max_graph_depth):
            self.image_draw(self.vertex_list[process_node], self.color_space[self.color_selection])
#            cv2.imshow('demo', self.img)
#            cv2.waitKey()
            
            self.color_selection = self.color_selection + 1
            
    def child_node_rectangle_draw(self, curr_node):
        node_draw_x = self.left_intervel_x + (self.children.index(curr_node) + 1)/(len(self.children) + 1) * self.intervel_x
        self.node_draw_x_list.append(node_draw_x)    
        node_draw_y = self.intervel_y * (self.graph.node[curr_node]['depth'][0] - 1) + self.node_draw_y_list[1]
        self.node_draw_y_list.append(node_draw_y)        
#        cv2.circle(self.img, (int(node_draw_x + self.image_width), int(node_draw_y)), self.node_radius, (255, 255, 255), -1)
#        
#        cv2.putText(self.img, 
#                    str(len(self.graph.node[curr_node]['guest'])), 
#                    (int(self.node_draw_x_list[curr_node] + self.image_width - self.node_radius/2) , int(self.node_draw_y_list[curr_node] + self.node_radius/2)), 
#                    self.font_parm, 0.8, (0,  0, 255), 1)
#        cv2.line(self.img, 
#                 (int(self.center_node_x + self.image_width), int(node_draw_y - self.intervel_y + self.node_radius)),
#                 (int(node_draw_x + self.image_width), int(node_draw_y - self.node_radius)),
#                  (255, 255, 255), 2)
        cv2.rectangle(self.img_nocolor,
                      (int(self.vertex_list[curr_node][0][0]), int(self.vertex_list[curr_node][0][1])),
                      (int(self.vertex_list[curr_node][1][0]), int(self.vertex_list[curr_node][1][1])),
                      (0, 0, 255), 3)
        
    def cal_cut_propotion(self, children):
        alpha = self.alpha_base * len(children)
        return np.random.dirichlet(alpha, size = 1)
    
    def make_propotion(self, curr_propotion):
        self.propotion = self.propotion + curr_propotion 
        self.propotion_tuple.append(self.propotion)
        self.propotion_list.append(self.propotion_tuple)
        self.propotion_tuple = [self.propotion]   
        
    def image_draw(self, vertex, color):   
        cv2.rectangle(self.img, (int(vertex[0][0]), int(vertex[0][1])), (int(vertex[1][0]), int(vertex[1][1])), color, -1)
        return self.img
        
    def __call__(self, all_tree_graph):
#        cv2.namedWindow('demo')
        map(self.nested_parse_vertex, all_tree_graph)
        
class write_tree_to_excel():
    def __init__(self, code_base):
        self.code_base = code_base        #code_base: decimal
        self.all_tree_code = []
        self.all_code_num = []
        
    def code_tree(self, tree_graph):
        self.code = 0
        self.graph = tree_graph
        self.code_list = [[ALL_GUEST_NUM]]
        self.to_process_node_id = [1]
        map(self.cal_code_list, self.to_process_node_id)
        map(self.map_code_list, self.code_list)
        if not self.code in self.all_tree_code:
            self.all_tree_code.append(self.code)
            self.all_code_num.append(1)
        else :
            index = self.all_tree_code.index(self.code)
            self.all_code_num[index] = self.all_code_num[index] + 1
            
    def cal_code_list(self, node_id):
        if len(self.code_list) == self.graph.node[node_id]['depth'][0]:
            self.code_list.append([])
            
        children = list(nx.DiGraph.successors(self.graph, node_id))
        if children:
            self.element_sorted_for = map(lambda x: (len(self.graph.node[x]['guest']), self.cal_children_single_guest_num(x)), children)
            self.element_no_repeat = list(set(self.element_sorted_for))
            self.element_no_repeat.sort()
            self.element_with_index_list = list(enumerate(self.element_sorted_for))
            self.element_sorted_for.sort()
            element_to_append = list(np.array(self.element_sorted_for)[:, 0])
            children_index_sorted = np.array(map(self.cal_index_sorted, self.element_no_repeat))
            children_index = filter(lambda x: x != None, children_index_sorted.flatten())
            children_sorted = map(lambda x: children[x], children_index)
            
            self.to_process_node_id.extend(children_sorted)
            self.code_list[self.graph.node[node_id]['depth'][0]].extend(element_to_append)

    def cal_children_single_guest_num(self, node_id):
        single_guest_num_list =  map(lambda x: len(self.graph.node[x]['guest']), nx.DiGraph.successors(self.graph, node_id))
        return len(single_guest_num_list) - single_guest_num_list.count(1) 
    
    def cal_index_sorted(self, element):
        self.target = element
        return map(self.find_index, range(len(self.element_with_index_list))) 
        
    def find_index(self, index):
        if self.element_with_index_list[index][1] == self.target:
            return index

    def map_code_list(self, code_list):
        print code_list
        map(self.code_principle, code_list)
        
    def code_principle(self, code_element):        
        self.code = long(self.code * self.code_base + code_element)
    
    def __call__(self, all_tree_graph):
        map(self.code_tree, all_tree_graph)
        tree_kind_num = {'tree_code': self.all_tree_code, 'code_num':self.all_code_num}    
        export_tree_kind_num = pd.DataFrame(tree_kind_num, columns = ['tree_code', 'code_num'], dtype='uint64')
        export_tree_kind_num.to_csv('tree_kind_num_' + str(ALL_GUEST_NUM) + '.csv')
        return tree_kind_num
    
def main():            
    One_Tree = Enumerate_tree(gamma = GAMMA, all_guest_num = ALL_GUEST_NUM)
    All_Tree = One_Tree(tree_num = TREE_NUM)
    Write = write_tree_to_excel(code_base = CODE_BASE)
    Ncrp_data = Write(All_Tree)
    One_image = Generate_image_by_tree(image_width = IMAGE_WIDTH, image_height = IMAGE_HEIGHT, alpha_base = ALPHA_BASE, color_space = COLOR_SPACE, graph_width = GRAPH_WIDTH)
    All_image = One_image(All_Tree)
    Parameters = {'gamma': GAMMA_RECORD, 'alpha':ALPHA_RECORD}    
    Export_Parameters = pd.DataFrame(Parameters, columns = ['gamma', 'alpha'])
    Export_Parameters.to_csv('parameter_record.csv')
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()
    
    
