# coding=utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt
img0 = cv2.imread('C:/Users/Johnhckuo/Desktop/104356024/img/bridge.jpg', 0) 
img1 = cv2.imread('C:/Users/Johnhckuo/Desktop/104356024/img/1.jpg', 0) 
img2 = cv2.imread('C:/Users/Johnhckuo/Desktop/104356024/img/2.jpg', 0) 
img3 = cv2.imread('C:/Users/Johnhckuo/Desktop/104356024/img/3.png', 0) 
height, width = img0.shape

result1 = 0
result2 = 0
result3 = 0

for i in range(height):
	for j in range(width):
		temp1 = int(img0[i][j]) - int(img1[i][j])
		temp2 = int(img0[i][j]) - int(img2[i][j])
		temp3 = int(img0[i][j]) - int(img3[i][j])
		result1 = result1 + (temp1*temp1)
		result2 = result2 + (temp2*temp2)
		result3 = result3 + (temp3*temp3)

result1 = float(result1)/(width*height)
result2 = float(result2)/(width*height)
result3 = float(result3)/(width*height)
		
print"MSE of image A : ",result1
print"MSE of image B : ",result2
print"MSE of image C : ",result3

