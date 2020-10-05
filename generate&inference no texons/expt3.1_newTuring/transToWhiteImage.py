import cv2
import numpy as np 

imageList = [95,409,545,634,695,991,1260,2067,2116,2240,2541,2656,2788,3007,3354,3395,3596,3799,4119,4510]

for imageIndex in imageList:
    imageOriginal = cv2.imread('../scenes/' + str(10000 + imageIndex) + '.png')
    blackPixelPosition = np.where((imageOriginal == [0, 0, 0]).all(axis = 2))
    imageOriginal[blackPixelPosition] = [255, 255, 255]
    cv2.imwrite('imagesWhite/' + str(10000 + imageIndex) + '.png', imageOriginal)
