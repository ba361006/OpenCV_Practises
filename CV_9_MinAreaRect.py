import cv2 
import numpy as np 


def getMinAreaRect(image_in):
    # return value of min_area_rect would be a tuple with value ((centre_x, centre_y), (width, height), angle)
    contours, _ = cv2.findContours(image_in, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    min_area_rect = cv2.minAreaRect(contours[0])
    return min_area_rect

def oriRect():
    # draw a rectangle 
    image_ori = np.zeros([500,500], dtype=np.uint8)
    raw, col = image_ori.shape
    image_ori[100:400, 200:300] = 255

    # and put a square with different colour on the rectangle to determine the direction
    image_ori[125:175, 225:275] = 100
    return image_ori

def rotate(image_in, rotate_angle):
    raw, col = image_in.shape
    min_area_rect = getMinAreaRect(image_in)

    rotation_matrix = cv2.getRotationMatrix2D(center = min_area_rect[0], 
                                              angle  = rotate_angle, 
                                              scale  = 1)
    image_rotated = cv2.warpAffine(image_in, rotation_matrix, (raw,col))
    return image_rotated

def rotateAndPutItBack(rotate_angle):
    print("### Start ###")
    image_ori         = oriRect()
    min_area_rect_ori = getMinAreaRect(image_in = image_ori)
    print("show the original image")
    print("min_area_rect of original image: ", min_area_rect_ori)
    cv2.imshow("image_ori", image_ori)
    cv2.waitKey()

    # create a 30 degrees rotated rectangle
    image_rotated_thirty  = rotate(image_in     = image_ori,
                                   rotate_angle = rotate_angle)
    min_area_rect_rotated = getMinAreaRect(image_in = image_rotated_thirty)
    print(f"\nmin_area_rect of {rotate_angle} degrees rotated image: ", min_area_rect_rotated)
    cv2.imshow(f"rotated_{rotate_angle}", image_rotated_thirty)
    cv2.waitKey()

    # turn the 30 degrees rotated rectangle back
    image_rotated_back = rotate(image_in     = image_rotated_thirty,
                                rotate_angle = -rotate_angle)
    min_area_rect_back = getMinAreaRect(image_in = image_rotated_back)
    print("\nmin_area_rect of rotated back image: ", min_area_rect_back)
    print("### End ###\n\n")
    cv2.imshow("rotated_back", image_rotated_back)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # building this file with opencv-contrib-python == 4.5.2.54
    # for further information, check README.md -> CV_9_MinAreaRect
    rotateAndPutItBack(rotate_angle = 30)
    rotateAndPutItBack(rotate_angle = -30)