import cv2 
import numpy as np 
from matplotlib import pyplot as plt 
import time


Image =  cv2.imread("./OpenCV/Image/water_coins.jpg")
Image_grey = cv2.cvtColor(Image, cv2.COLOR_RGB2GRAY)
_, Threshold = cv2.threshold(Image_grey, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# cv2.imshow("Threshold", Threshold)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# Morphologizing
Kernel = np.ones([3,3], dtype = np.uint8)
Open = cv2.morphologyEx(Threshold, cv2.MORPH_OPEN, Kernel, iterations = 2)
Sure_Background = cv2.dilate(Open, Kernel, iterations = 3)
# cv2.imshow("Open", Open)
# cv2.imshow("Sure_Background", Sure_Background)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.distanceTransform is to compute the Euclidean Distance(If use cv2.DIST_L2) from each nonzero pixel to zero pixel
# the longer the distance is, the higher the return value is
Dist_Transform = cv2.distanceTransform(Open, cv2.DIST_L2, 3)
_, Sure_fg = cv2.threshold(Dist_Transform, 0.7*Dist_Transform.max(), 255, 0)
Sure_fg = np.uint8(Sure_fg)
# cv2.imshow("Dist_Transform", np.uint8(Dist_Transform)*10)
# cv2.imshow("Sure_fg", Sure_fg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


unknown = cv2.subtract(Sure_Background, Sure_fg)
# cv2.imshow("unknown", unknown)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.connectedComponents is to give each isolated object a value which is from 0 ~ N-1(N is the number of isolated object)
_, Markers = cv2.connectedComponents(Sure_fg)

# Watershed method will consider the place where the intensity of pixel is 0 as an unknown area, so here we plus 1 to avoid the problem
# and let the pixels located in the unknown area become 0
Markers = Markers + 1
Markers[unknown==255] = 0

# Implement one the variants of watershed which will return -1 to the boundaries between the objects
Markers = cv2.watershed(Image, Markers)
Image[Markers == -1] = [0, 0, 255]
cv2.imshow("Results", Image)
cv2.waitKey(0)
cv2.destroyAllWindows()


