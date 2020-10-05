import cv2
import os
import numpy as np 

project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'image')
IMAGE_ORDER = list(np.array([5, 18, 42, 77, 157, 168, 183, 369, 407, 447, 587, 619, \
        632, 653, 660, 702, 732, 750, 842, 864]) + 10000)
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 768
 
def main():
    for i in range(len(IMAGE_ORDER)):
        originImageName = os.path.join(project_path, str(IMAGE_ORDER[i])+'.png')
        originImage = cv2.imread(originImageName)
        cv2.imwrite(os.path.join(project_path,'0_'+str(IMAGE_ORDER[i])+'.png'), originImage) 

if __name__ == '__main__':
    main()
