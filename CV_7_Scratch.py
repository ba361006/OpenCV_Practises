import cv2
import numpy as np 

# input image
image_ori = cv2.imread("./Image/Scratch.png")
image_grey = cv2.cvtColor(image_ori, cv2.COLOR_RGB2GRAY)

# get sobel image
blur = cv2.GaussianBlur(image_grey, (15,15), 5)
x = cv2.Sobel(blur, cv2.CV_16S, 1, 0)
y = cv2.Sobel(blur, cv2.CV_16S, 0, 1)
absx = cv2.convertScaleAbs(x)
absy = cv2.convertScaleAbs(y)
Sobel = cv2.addWeighted(absx, 0.5, absy, 0.5, 0)
Sobel = cv2.normalize(Sobel, None, 0, 255, cv2.NORM_MINMAX)

# morphologizing 
thresh_bin, _ = tools.Threshold(Sobel, 72, 255)
kernel = np.ones([3,3], dtype = np.int)
closing = cv2.morphologyEx(thresh_bin, cv2.MORPH_CLOSE, kernel)

# get region of scratch by finding the contour which has more than 2000 pixels
ret, labels = cv2.connectedComponents(closing)
for label in range(1, ret):
    # # to visualize the process, please uncomment here
    # visualize = np.array(labels, dtype = np.uint8)
    # visualize[labels == label] = 255
    # cv2.imshow("visualize", visualize)
    # cv2.waitKey()
    
    mask = np.array(labels)
    area = len(np.argwhere(mask[labels == label]))

    if area > 2000:
        palette = np.zeros_like(mask, dtype = np.uint8)
        palette[labels == label] = 255
        cv2.imshow("compoent", palette)
        cv2.waitKey()
        contours, _ = cv2.findContours(palette, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(image_ori, contours, -1, (0,255,0))



cv2.imshow("image_ori", image_ori)
cv2.imshow("image_grey", image_grey)
cv2.imshow("blur", blur)
cv2.imshow("Sobel", Sobel)
cv2.imshow("thresh_bin", thresh_bin)
cv2.imshow("closing", closing)
cv2.waitKey()

