# coding=utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt

def test(arr, h, w):

	sort = [];
	prime = [];
	targetIndex = [0]*h;

	#建立空的二維陣列，用於儲存前1/4大的係數位置
	for i in range(h):
		targetIndex[i] = [0]*w;
		
	#取絕對值，才能比較大小
	fshift1 = np.abs(arr)
	
	#將二維陣列轉為一維，才能進行排序
	for i in fshift1:
		for j in i:
			sort.append(j);
			
	#儲存排序前的陣列
	prime = sort;

	#開始排序
	sort = sorted(sort, reverse=True)
	
	#將結果(一維陣列)存進result中
	result = [];
	for i in range(h*w/4):
		result.append(sort[i])

	#讀取原始二維陣列中的元素，查詢result中前1/4大的元素在二維陣列中的實際位置，並將位置存進targetIndex
	for i in range(h):
		for j in range(w):
			if (fshift1[i][j] in result):
				targetIndex[i][j] = 1;
			
	#根據剛剛存的位置，若不存在於targetIndex中的元素，就把該係數設為0，反之則保留
	for i in range(h):
		for j in range(w):
			if (targetIndex[i][j] == 0):
				arr[i][j] = 0;
		
img = cv2.imread('C:/Users/Johnhckuo/Desktop/104356024/img/bridge.jpg',0) 
height, width = img.shape
f = np.fft.fft2(img)
test(f, height, width)

#將結果顯示出來
plt.subplot(121),plt.imshow(img,'gray'),plt.title('original')
img_back = np.fft.ifft2(f)
img_back = np.abs(img_back)

cv2.imwrite('C:/Users/Johnhckuo/Desktop/104356024/img/1.jpg',img_back)
plt.subplot(122),plt.imshow(img_back,'gray'),plt.title('after')
plt.xticks([]),plt.yticks([])

plt.show()
cv2.waitKey(0);
cv2.destroyAllWindows();

