# coding=utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

#Zero-Padding函式
def zeropad2(x, shape):
	m, n= x.shape
	p, q = shape
	assert p > m
	assert q > n
	tb = np.zeros(((p - m) / 2, n))
	lr = np.zeros((p, (q - n) / 2))
	x = np.append(tb, x, axis = 0)
	x = np.append(x, tb, axis = 0)
	x = np.append(lr, x, axis = 1)
	x = np.append(x, lr, axis = 1)
	return x
	
def reshape((x, y), factor):
    return (int(factor * x), int(factor * y))

img = cv2.imread('C:/Users/Johnhckuo/Desktop/104356024/img/bridge.jpg', 0) 
height, width = img.shape

#Nsamll為要縮小成的圖形大小
#Nbig為原本的大小
Nsmall = 128
Nbig = 256

#將圖片averaging為128x128
resized = img.reshape([Nsmall, Nbig/Nsmall, Nsmall, Nbig/Nsmall]).mean(3).mean(1)

#傅利葉轉換後，再進行Zero-Padding
fft = np.fft.fft2(resized)
fshift = np.fft.fftshift(fft)

h, w = resized.shape

#測試直接將128x128的放大兩倍，用來和zero-padding後的結果做比較
fshift2 = cv2.resize(resized,(2*w, 2*h), interpolation = cv2.INTER_CUBIC)
#將圖片Zero-Padding為原圖大小: 256x256(最後一個參數為指定圖片放大倍率)
fshift = zeropad2(fshift, reshape(fshift.shape, 2.0))

#逆向傅利葉轉換，並且把結果印出來
f1shift = np.fft.ifftshift(fshift)
ifft = np.fft.ifft2(f1shift)

cv2.imshow("original", img)
ifft = np.real(ifft)

#將結果存為3.png，以供之後比較MSE使用
fig2 = plt.figure(figsize = ifft.shape, dpi = 1)
fig2.figimage(ifft, cmap = plt.cm.gray)
plt.savefig('C:/Users/Johnhckuo/Desktop/104356024/img/3.png', dpi = 1)

#將直接放打兩倍的結果存為3(scale).png，並顯示出來和3.png做比較
fig2 = plt.figure(figsize = fshift2.shape, dpi = 1)
fig2.figimage(fshift2, cmap = plt.cm.gray)
plt.savefig('C:/Users/Johnhckuo/Desktop/104356024/img/3(scale).png', dpi = 1)

plt.show()
cv2.waitKey(0);
cv2.destroyAllWindows();

