import cv2
import numpy as np 
from matplotlib import pyplot as plt



# cap = cv2.VideoCapture(0)

# while(True):
#     _, frame = cap.read()
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#     lower = np.array([80,0,0])
#     upper = np.array([200,255,255])

#     mask = cv2.inRange(hsv, lower, upper)
#     res = cv2.bitwise_and(frame , frame, mask = mask)

#     cv2.imshow('frame',frame)
#     cv2.imshow('mask', mask)
#     cv2.imshow('res',res)

#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break

# cv2.destroyAllWindows()
# cap.release()

dustmite = cv2.imread('./OpenCV/imgs/dustmite3.jpg')
hsv = cv2.cvtColor(dustmite, cv2.COLOR_BGR2HSV)
a = cv2.cvtColor(dustmite,cv2.COLOR_BGR2RGB)
plt.subplot(3,4,1)
plt.imshow(a,'gray')
title = []
# for i in range(8):
#     lower = np.array([50 - 10*i,0,0])
#     upper = np.array([80,255,255])

#     title.append(str((50 - 10*i, 80)))
#     mask = cv2.inRange(hsv, lower,upper)
#     res = cv2.bitwise_and(dustmite,dustmite, mask = mask)

#     plt.subplot(3,4,i+2)
#     plt.imshow(res)
#     plt.title(title[i])
# plt.show()
cv2.namedWindow('Ori',0)
cv2.resizeWindow('Ori',640, 480)
cv2.imshow('Ori',dustmite)

cv2.namedWindow('output',0)
cv2.resizeWindow('output',640, 480)

for i in range(8):
    lower = np.array([0,0,0])
    upper = np.array([80,255,255])
    mask = cv2.inRange(hsv, lower,upper)
    res = cv2.bitwise_and(dustmite,dustmite, mask = mask)
    cv2.imshow('output',res)
    cv2.waitKey(0)

cv2.destroyAllWindows()