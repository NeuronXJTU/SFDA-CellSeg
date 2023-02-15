import numpy as np
import cv2
import os


img = []

path_result = "./data/result/eval/"
path_origin = "./data/img_evaluate/"
path_canny = "./data/canny/"
out_name = '0'
list1 = os.listdir(path_result)

for i in list1:

    img_dir = path_result + i

    j = i[2:]
    print(j[:-7])
    img_ori = path_origin + j[:-7] + '.png'
    img_ori = cv2.imread(img_ori)

    img_ori = cv2.resize(img_ori,(512,512))

    out_name = path_canny + i
    img = cv2.imread(img_dir)
    img2 = cv2.Canny(img, 90, 200)
    img2[img2 > 60] = 127
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    dilate = cv2.dilate(img2, kernel, iterations=2)
    img_ori[dilate > 60] = (117, 250, 76)#127


    cv2.imwrite(out_name + '.png',img_ori)

    img = []