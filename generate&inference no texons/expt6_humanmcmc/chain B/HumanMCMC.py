# -*- coding: utf-8 -*-
"""
Created on Fri Nov 03 17:40:18 2017

@author: Edward Coen
"""

from __future__ import division
import numpy as np
import os
import cv2

subject_num = int(input("subject_num: "))
project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'image')
pub_path = os.path.join(project_path, 'pub')

IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 768
IMAGE_ORDER = [10005, 10018, 10042, 10077, 10157, 10168, 10183, 10369, 10407, 10447, 10587, 10619, 10632, 10653, 10660, 10702, 10732, 10750, 10842, 10864]

CLICK_NUMBER = 0
CLICK_POINTS = []
POSSIBLE_LINES_POINTS_LIST = []

def mouse_select(event, x, y, flag, param):
    global subject_num, CLICK_NUMBER, CLICK_POINTS, CONFIRM_POINTS, COLORFILL_NUMBER, CLICK_COLOR, POSSIBLE_LINES_POINTS_LIST
    if event == cv2.EVENT_LBUTTONDOWN:
        if CLICK_NUMBER % 2 == 0 and CLICK_NUMBER < 10:
            if list(param[0][y][x]) == [0, 0, 255]:
                if len(CLICK_POINTS) == 2:
                    cv2.circle(param[0], (int(CLICK_POINTS[0][0]), int(CLICK_POINTS[0][1])), 1, (0, 0, 255), -1)
                    cv2.circle(param[0], (int(CLICK_POINTS[1][0]), int(CLICK_POINTS[1][1])), 1, (0, 0, 255), -1)
                    for i in range(len(POSSIBLE_LINES_POINTS_LIST)//2):        
                        cv2.line(param[0], (int(POSSIBLE_LINES_POINTS_LIST[i * 2][0]), int(POSSIBLE_LINES_POINTS_LIST[i * 2][1])), (int(POSSIBLE_LINES_POINTS_LIST[i * 2 + 1][0]),
                            int(POSSIBLE_LINES_POINTS_LIST[i * 2 + 1][1])), (0, 0, 255), 2)
                    cv2.imshow('image', param[0])
                    del CLICK_POINTS[:]
                    del POSSIBLE_LINES_POINTS_LIST[:]
                    cv2.imwrite(os.path.join(project_path, str(subject_num)+'_'+str(IMAGE_ORDER[param[1]])+'_'+str(((CLICK_NUMBER - 1)//2)%6)+'.png'), param[0])
                CLICK_POINTS.append([x, y])
                POSSIBLE_LINES_POINTS_LIST = get_possible_lines(x, y, param[0])
                for i in range(len(POSSIBLE_LINES_POINTS_LIST)//2):
                    cv2.line(param[0], (int(POSSIBLE_LINES_POINTS_LIST[i * 2][0]), int(POSSIBLE_LINES_POINTS_LIST[i * 2][1])), (int(POSSIBLE_LINES_POINTS_LIST[i * 2 + 1][0]),
                        int(POSSIBLE_LINES_POINTS_LIST[i * 2 + 1][1])), (0, 255, 0), 2)
                cv2.imshow('image', param[0])
                cv2.circle(param[0], (int(CLICK_POINTS[0][0]), int(CLICK_POINTS[0][1])), 1, (255, 0, 0), -1)
                cv2.imshow('image', param[0])
                CLICK_NUMBER = CLICK_NUMBER + 1
                
        if CLICK_NUMBER % 2 == 1 and CLICK_NUMBER <= 10:
            if list(param[0][y][x]) == [0, 255, 0]: 
                if abs(x - CLICK_POINTS[0][0]) <= 20:
                    CLICK_POINTS.append([x, y])
                    cv2.circle(param[0], (int(CLICK_POINTS[1][0]), int(CLICK_POINTS[1][1])), 1, (255, 0, 0), -1)
                    cv2.imshow('image', param[0])
                    cv2.line(param[0], (int(np.average(CLICK_POINTS, axis = 0)[0]), int(CLICK_POINTS[0][1])), (int(np.average(CLICK_POINTS, axis = 0)[0]), int(CLICK_POINTS[1][1])),
                            (0, 0, 255), 4)
                    cv2.imshow('image', param[0])
                    CLICK_NUMBER = CLICK_NUMBER + 1
                    
                if abs(y - CLICK_POINTS[0][1]) <= 20:
                    CLICK_POINTS.append([x, y])
                    cv2.circle(param[0], (int(CLICK_POINTS[1][0]), int(CLICK_POINTS[1][1])), 1, (255, 0, 0), -1)
                    cv2.imshow('image', param[0])
                    cv2.line(param[0], (int(CLICK_POINTS[0][0]), int(np.average(CLICK_POINTS, axis = 0)[1])), (int(CLICK_POINTS[1][0]), int(np.average(CLICK_POINTS, axis = 0)[1])),
                            (0, 0, 255), 4)
                    cv2.imshow('image', param[0])
                    CLICK_NUMBER = CLICK_NUMBER + 1
                
                
                    
            if CLICK_NUMBER == 10:
                cv2.circle(param[0], (int(CLICK_POINTS[0][0]), int(CLICK_POINTS[0][1])), 1, (0, 0, 255), -1)
                cv2.circle(param[0], (int(CLICK_POINTS[1][0]), int(CLICK_POINTS[1][1])), 1, (0, 0, 255), -1)
                for i in range(len(POSSIBLE_LINES_POINTS_LIST)//2):        
                    cv2.line(param[0], (int(POSSIBLE_LINES_POINTS_LIST[i * 2][0]), int(POSSIBLE_LINES_POINTS_LIST[i * 2][1])), (int(POSSIBLE_LINES_POINTS_LIST[i * 2 + 1][0]),
                        int(POSSIBLE_LINES_POINTS_LIST[i * 2 + 1][1])), (0, 0, 255), 2)
                cv2.imshow('image', param[0])
                del POSSIBLE_LINES_POINTS_LIST[:]
                cv2.imwrite(os.path.join(project_path, str(subject_num)+'_'+str(IMAGE_ORDER[param[1]])+'_'+str(((CLICK_NUMBER - 1)//2)%6)+'.png'), param[0])
                press_image = cv2.imread(os.path.join(pub_path, 'press1.png'))
                param[0][68:100, 428:600] = press_image
                cv2.imshow('image', param[0])
                
def get_possible_lines(x, y, img):
    possible_points = []
    if (list(img[y - 5][x]) == [0, 0, 255]) or (list(img[y + 5][x]) == [0, 0, 255]):
        for i in range(x + 5, int(0.8*IMAGE_WIDTH)):
            if list(img[y][i]) == [0, 0, 255]:
                possible_points.extend([[i + 1, y - 15], [i + 1, y + 15]])
                break
        
        for i in range(5, x):
            if list(img[y][x - i]) == [0, 0, 255]:
                possible_points.extend([[x - i - 1, y - 15], [x - i - 1, y + 15]])
                break
            
    if (list(img[y][x - 5]) == [0, 0, 255]) or (list(img[y][x + 5]) == [0, 0, 255]):
        for i in range(y + 5, int(0.8*IMAGE_HEIGHT)):
            if list(img[i][x]) == [0, 0, 255]:
                possible_points.extend([[x - 15, i + 1], [x + 15, i + 1]])
                break
        
        for i in range(5, y):
            if list(img[y - i][x]) == [0, 0, 255]:
                possible_points.extend([[x - 15, y - i - 1], [x + 15, y - i - 1]])
                break
    return possible_points

def iamge_generate():
    global CLICK_NUMBER, CLICK_POINTS, subject_num
    np.random.shuffle(IMAGE_ORDER)
    i = 0

    cv2.namedWindow('image')
    image = np.zeros([IMAGE_HEIGHT, IMAGE_WIDTH, 3], 'uint8') 
    image[:] = 40
    press_image = cv2.imread(os.path.join(pub_path, 'press1.png'))
    image[68:100, 428:600] = press_image
    cv2.imshow('image', image)
    cv2.waitKey()

    while i < len(IMAGE_ORDER):
        
        image = np.zeros([IMAGE_HEIGHT, IMAGE_WIDTH, 3], 'uint8') 
        image[:] = 40
        cv2.imshow('image', image)
        blankTimeBeforeMemory = np.random.uniform(200, 300)
        cv2.waitKey(int(blankTimeBeforeMemory))
        
        if subject_num == 1:
            originImageName = os.path.join(project_path, str(IMAGE_ORDER[i])+'.png')
            originImage = cv2.imread(originImageName)
            scale_percent = 60 # percent of original size
            width = int(IMAGE_WIDTH * scale_percent / 100)
            height = int(IMAGE_HEIGHT * scale_percent / 100)
            dim = (width, height)
            perceptionImage = cv2.resize(originImage, dim, interpolation = cv2.INTER_AREA)
            image[int(0.2*IMAGE_HEIGHT):int(0.8*IMAGE_HEIGHT) - 1, int(0.2*IMAGE_WIDTH):int(0.8*IMAGE_WIDTH) - 1] = perceptionImage
        else:
            perceptionImageName = os.path.join(project_path, str(subject_num - 1)+'_'+str(IMAGE_ORDER[i])+'.png')
            perceptionImage = cv2.imread(perceptionImageName)
            image = perceptionImage
            
        cv2.imshow("image", image)
        cv2.waitKey(2000)
        
        image = np.zeros([IMAGE_HEIGHT, IMAGE_WIDTH, 3], 'uint8') 
        image[...] = 40
        cv2.imshow('image', image)
        cv2.waitKey(500)
        
        CLICK_NUMBER = 0
        CLICK_POINTS = []
        image = np.zeros([IMAGE_HEIGHT, IMAGE_WIDTH, 3], 'uint8') 
        image[...] = 40
        cv2.imshow('image', image)
        CLICK_NUMBER = 0
        
        ###redraw
        image = np.zeros([IMAGE_HEIGHT, IMAGE_WIDTH, 3], 'uint8') 
        image[:] = 40
        strat_vertex = [[0.2*IMAGE_WIDTH, 0.2*IMAGE_HEIGHT], [0.8*IMAGE_WIDTH, 0.2*IMAGE_HEIGHT], [0.8*IMAGE_WIDTH, 0.8*IMAGE_HEIGHT], [0.2*IMAGE_WIDTH, 0.8*IMAGE_HEIGHT]]
        cv2.rectangle(image, (int(strat_vertex[0][0]), int(strat_vertex[0][1])), (int(strat_vertex[2][0]), int(strat_vertex[2][1])), (0, 0, 0), -1)
        cv2.imshow('image', image)
        cv2.polylines(image, [np.array(strat_vertex, np.int32)], True, (0, 0, 255), 4)
        cv2.imshow('image', image) 
        cv2.setMouseCallback('image', mouse_select, [image, i])
        key = cv2.waitKey()
        image[68:100, 428:600] = 40 
        cv2.imshow('image', image)
        
        cv2.imwrite(os.path.join(project_path, str(subject_num)+'_'+str(IMAGE_ORDER[i])+'.png'), image) 
        
        if CLICK_NUMBER == 10:
            i = i + 1
        if key == 27:
            break

    cv2.destroyAllWindows()

def main():
    iamge_generate()

if __name__ == '__main__':
    main()
    
