import numpy as np
import cv2
import os
from math import sin,cos,radians
from PIL import Image
import torchvision
from math import sin,cos,radians
from utils import get_pixel,rotate,get_neighbors
from collections import defaultdict
import queue
import copy
import time
import imutils
import time

show_img_rgb = lambda img: Image.fromarray(img.astype(np.uint8),"RGB")
show_img_L = lambda img: Image.fromarray(img.astype(np.uint8),"L")

# rotate the object by 330 degerees 
angle = 330

def rotate(image, angle, center = None, scale = 1.0): 
    image = np.asarray(image)
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated

# load original image, silouette, and background
test_orig = np.asarray(Image.open("./materials/2008_001971_rgb.png"))
test_sil = np.asarray(Image.open("./materials/2008_001971_silhouette.png"))
test_background = np.asarray(Image.open("./materials/2008_001971.png"))

# delete the padding in the original image
right = None
bottom = None
for i in reversed(range(480)):
    if(i<460):
        if(right is None and np.any(test_background[20,i,:]!=0)):
            right = i
        if(bottom is None and np.any(test_background[i,20,:]!=0)):
            bottom = i
            
good_test_background = test_background[:bottom,:right]
l,t,r,b = Image.fromarray(test_sil).getbbox()
mid_point = (l + r) //2, (b+t) // 2 

# rotate according to the angle, 330 in this example
mask_rotate = rotate(test_sil,angle,mid_point)
obj_rotate = rotate(test_orig,angle,mid_point)
indices = np.where(mask_rotate != 0)
background = good_test_background.copy()
background[indices] = obj_rotate[indices]
show_img_rgb(background).save(os.path.join("./output","rotation_"+str(angle)+".png"),"png")