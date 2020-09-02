import cv2
import numpy as np 

img1  = cv2.imread('./Image/lena.png', cv2.IMREAD_COLOR)
img2 = cv2.imread('./Image/python.png', cv2.IMREAD_COLOR)
cv2.imshow('Ori_lena',img1)
cv2.imshow('Ori_python', img2)



img22gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
_, py_mask = cv2.threshold(img22gray, 100, 255, cv2.THRESH_BINARY_INV)
# cv2.imshow('py_mask',py_mask)
mask2bgr = cv2.cvtColor(py_mask, cv2.COLOR_GRAY2BGR)
python =  cv2.add(mask2bgr,img2)
cv2.imshow('python',python)
# cv2.imshow('img1',img1)



python2gray = cv2.cvtColor(python,cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(python2gray, 200, 255, cv2.THRESH_BINARY_INV)
# cv2.imshow('mask',mask)
mask_inv = cv2.bitwise_not(mask)

bg = cv2.resize(cv2.bitwise_and(python,python, mask = mask_inv),(100,100))
# cv2.imshow('bg',bg)
fg = cv2.resize(cv2.bitwise_and(python,python, mask = mask),(100,100))
# cv2.imshow('fg',fg)

rows, cols, channels = img1.shape
output = img1[0:rows, 0:cols, 0:channels]
knockout = cv2.bitwise_and(output[0:100,0:100],bg)
output[0:100,0:100] = cv2.add(knockout,fg)

cv2.imshow('output',output)
cv2.waitKey(0)
cv2.destroyAllWindows()