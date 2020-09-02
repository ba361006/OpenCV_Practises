"""
Function Name: tools
Description: Collecting some commonly used functions in Image Processing
Argument: None
Parameter: None
Return: None
Edited by: 2020-08-20 Bill Gao
"""
import numpy as np 
import cv2


def thresholdTest(input_image, bgr = False):
    """
    Function Name: thresholdTest
    
    Description: input an image, and select the upper and lower threshold value
    
    Argument: 
              input_image [np.array] -> input image
              bgr [bool] -> Convert a bgr image into a greyscale image
              
    Parameters: None
    
    Return: 
            
    Edited by: 2020-08-20 Bill Gao
    """    
    if len(input_image.shape) == 3:
        if bgr == True:
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        else:
            input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)

    cv2.namedWindow("Trackbars")
    cv2.createTrackbar("upper", "Trackbars", 255, 255, lambda tmp:None)
    cv2.createTrackbar("lower", "Trackbars", 0, 255, lambda tmp:None)
    while (True):
        upper = cv2.getTrackbarPos("upper", "Trackbars")
        lower = cv2.getTrackbarPos("lower", "Trackbars")
        binary_image = cv2.inRange(input_image, lower, upper)
        bitwise_image = cv2.bitwise_and(input_image, input_image, mask = binary_image)
        cv2.imshow("binary_image", binary_image)
        cv2.imshow("bitwise_image", bitwise_image)

        if cv2.waitKey(1) & 0xFF == ord("w"):
            print("lower, upper = {}, {}\n".format(lower, upper))
            break
    cv2.destroyWindow("Trackbars")
    cv2.destroyWindow("binary_image")
    cv2.destroyWindow("bitwise_image")



def threshold(input_image, lower, upper, show_image = None, bgr = False):
    """
    Function Name: threshold
    
    Description: Convert the input image into a hsv image, and turn pixels whose intensity are between
                 the given lower and upper threshold value into white, and turn the other into black.
    
    Argument: 
              input_image [np.array] -> Input Image
              lower [int] -> lower threshold value 
              upper [int] -> upper threshold value 
              show_image [bool] -> Show Result
              bgr [bool] -> Convert a bgr image into a greyscale image
              
    Parameters: None
    
    Return: 
            [np.array] -> Binary image
            [np.array] -> Colour image
           
    Edited by: 2020-08-20 Bill Gao
    """
    if len(input_image.shape) == 3:
        if bgr == True:
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        else:
            input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)

    binary_image = cv2.inRange(input_image, lower, upper)
    bitwise_image = cv2.bitwise_and(input_image, input_image, mask = binary_image)
    
    if show_image:
        cv2.imshow("threshold_out_binary", binary_image)
        cv2.imshow("threshold_out_colour", bitwise_image)
        cv2.waitKey(0)
        cv2.destroyWindow("threshold_out_binary")
        cv2.destroyWindow("threshold_out_colour")
    return binary_image, bitwise_image



def hsvTest(input_image, bgr = False):
    """
    Function Name: hsvTest
    
    Description: input an image, and select the upper and lower threshold value of HSV
                 
    
    Argument: 
              input_image [np.array] -> input image
              bgr [bool] -> Convert a bgr image into a greyscale image
    Parameters: None
    
    Return: 
            
    Edited by: 2020-08-20 Bill Gao
    """    
    if bgr == True:
        img_hsv = cv2.cvtColor(input_image, cv2.COLOR_bgr2HSV)
    else:
        img_hsv = cv2.cvtColor(input_image, cv2.COLOR_RGB2HSV)

    cv2.namedWindow("Trackbars")
    cv2.createTrackbar("upper_H", "Trackbars", 180, 180, lambda tmp:None)
    cv2.createTrackbar("upper_S", "Trackbars", 255, 255, lambda tmp:None)
    cv2.createTrackbar("upper_V", "Trackbars", 255, 255, lambda tmp:None)
    cv2.createTrackbar("lower_H", "Trackbars", 0, 180, lambda tmp:None)
    cv2.createTrackbar("lower_S", "Trackbars", 0, 255, lambda tmp:None)
    cv2.createTrackbar("lower_V", "Trackbars", 0, 255, lambda tmp:None)
    print("Press 'w' to end the while loop")

    while (True):
        upper_H = cv2.getTrackbarPos("upper_H", "Trackbars")
        upper_S = cv2.getTrackbarPos("upper_S", "Trackbars")
        upper_V = cv2.getTrackbarPos("upper_V", "Trackbars")
        lower_H = cv2.getTrackbarPos("lower_H", "Trackbars")
        lower_S = cv2.getTrackbarPos("lower_S", "Trackbars")
        lower_V = cv2.getTrackbarPos("lower_V", "Trackbars")

        lower = np.array([lower_H, lower_S, lower_V], dtype = np.uint8)
        upper = np.array([upper_H, upper_S, upper_V], dtype = np.uint8) 
        binary_image = cv2.inRange(img_hsv,lower ,upper )
        bitwise_image = cv2.bitwise_and(input_image, input_image, mask = binary_image)
        cv2.imshow("binary_image", binary_image)
        cv2.imshow("bitwise_image", bitwise_image)

        if cv2.waitKey(1) & 0xFF == ord("w"):
            print("lower, upper = [{},{},{}], [{},{},{}]\n".format(lower[0],
                                                                   lower[1],
                                                                   lower[2],
                                                                   upper[0],
                                                                   upper[1],
                                                                   upper[2]))
            break
    cv2.destroyWindow("binary_image")
    cv2.destroyWindow("bitwise_image")
    cv2.destroyWindow("Trackbars")



def hsv(input_image, lower, upper, show_image = None, bgr = False):
    """
    Function Name: hsv
    
    Description: Convert the input image into a hsv image, and turn pixels whose intensity are between
                 the given lower and upper threshold value into white, and turn the other into black.
    
    Argument: 
              input_image [np.array] -> Input Image
              lower [list] -> lower threshold value
              upper [list] -> upper threshold value
              show_image [bool] -> Show Result
              bgr [bool] -> Convert a bgr image into a hsv image
              
    Parameters: None
    
    Return: 
            [np.array] -> Binary image
            [np.array] -> Colour image
           
    Edited by: 2020-08-20 Bill Gao
    """    
    if bgr == True:
        hsv_image = cv2.cvtColor(input_image, cv2.COLOR_bgr2HSV)
        lower = np.array(lower, dtype = np.uint8)
        upper = np.array(upper, dtype = np.uint8)
    else:
        hsv_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2HSV)
        lower = np.array(lower, dtype = np.uint8)
        upper = np.array(upper, dtype = np.uint8)

    
    binary_image = cv2.inRange(hsv_image, lower, upper)
    bitwise_image = cv2.bitwise_and(input_image, input_image, mask = binary_image)

    if show_image:
        cv2.imshow("binary_image", binary_image)
        cv2.imshow("bitwise_image", bitwise_image)
        cv2.waitKey(0)
        cv2.destroyWindow("binary_image")
        cv2.destroyWindow("bitwise_image")
    return binary_image, bitwise_image
        

def cannyTest(input_image, bgr = False):
    """
    Function Name: cannyTest
    
    Description: input an image, and select the upper and lower threshold value of canny edge detection 
    
    Argument: 
              input_image [np.array] -> input image
              bgr [bool] -> Convert a bgr image into a greyscale image
              
    Parameters: None
    
    Return: 
            
    Edited by: 2020-08-20 Bill Gao
    """    
    if bgr == True:
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    else:
        input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)


    blurred = cv2.GaussianBlur(input_image, (3,3), 0)
    cv2.namedWindow("Trackbars")
    cv2.createTrackbar("upper", "Trackbars", 0, 255, lambda tmp:None)
    cv2.createTrackbar("lower", "Trackbars", 0, 255, lambda tmp:None)
    print("\n Press 'w' to end the while loop")
    while (True):
        upper = cv2.getTrackbarPos("upper", "Trackbars")
        lower = cv2.getTrackbarPos("lower", "Trackbars")
        canny_image = cv2.Canny(input_image , lower, upper)
        cv2.imshow("canny_image", canny_image)

        if cv2.waitKey(1) & 0xFF == ord("w"):
            print("[lower, upper] = [{},{}]".format(lower, upper))
            break
    cv2.destroyWindow("canny_image")


def Canny(input_image,lower = 0, upper = 255, bgr = False):
    """
    Function Name: Canny
    
    Description: input lower and upper threshold value to apply canny edge detection
    
    Argument: 
              input_image [type] -> [description]
              bgr [bool] -> Convert a bgr image into a greyscale image
              
    Parameters: None
    
    Return: 
            [np.arrary] -> canny edge image
           
    Edited by: 2020-08-20 Bill Gao
    """    
    if bgr == True:
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    else:
        input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(input_image, (3,3), 0)
    canny_image = cv2.Canny(blurred, lower, upper)
    return canny_image


# Image_extracting Parameters
mouse_row_0, mouse_col_0, mouse_row_1, mouse_col_1, mouse_count = 0, 0, 0, 0, 0

def mouseAction(event, x, y, flags, param):
    global mouse_row_0, mouse_col_0, mouse_row_1, mouse_col_1, mouse_count
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Left button down")
        if mouse_count == 0:
            mouse_col_0, mouse_row_0 = x, y
            mouse_count = 1
        else:
            mouse_col_1, mouse_row_1 = x, y
            mouse_count = 0
        
def Image_extracting(input_image, return_image = True):
    global mouse_row_0, mouse_col_0, mouse_row_1, mouse_col_1, mouse_count
    
    cv2.namedWindow("Image_extracting")

    cv2.imshow("Image_extracting", input_image)
    cv2.setMouseCallback("Image_extracting", mouseAction)
    cv2.waitKey(0)
    
    extract_img = input_image[mouse_row_0: mouse_row_1, mouse_col_0:mouse_col_1]
    cv2.imshow("extract", extract_img)
    cv2.waitKey(0)
    cv2.destroyWindow("extract")

    if return_image:
        return extract_img



def slidingWindow(input_image, kernel):
    """
    Function Name: slidingWindow
    
    Description: Apply the sliding window method without using opencv
    
    Argument: 
              input_image [np.array] -> input image
              kernel [np.array] -> kernel for applying the sliding window method
              
    Parameters: None
    
    Return: 
            [np.array] -> processed image
           
    Edited by: 2020-08-20 Bill Gao
    """    
    row_image, col_image, channel_image = input_image.shape
    k_row, k_col = kernel.shape

    feature_row = row_image - k_row + 1
    feature_col = col_image - k_col + 1
    feature = np.lib.stride_tricks.as_strided(input_image,
                                              shape = (1, 
                                                       channel_image, 
                                                       feature_row, 
                                                       feature_col, 
                                                       k_row, 
                                                       k_col),
                                              strides = (1, 
                                                         1, 
                                                         channel_image*col_image, 
                                                         channel_image*1, 
                                                         channel_image*col_image, 
                                                         channel_image*1))

    feature = feature.reshape(feature_row*feature_col*3, k_row, k_col)
    feature_map = np.uint8(np.tensordot(kernel, feature, [(0,1), (1,2)]))

    Processed_image = np.lib.stride_tricks.as_strided(feature_map,
                                                      shape = (feature_row,
                                                               feature_col,
                                                               channel_image),
                                                      strides = (feature_col,
                                                                 1,
                                                                 feature_row * feature_col))

    return Processed_image

def cameraFocusTest(Port):
    """
    Function Name: cameraFocusTest
    
    Description: Adjust the foucs of camera
    
    Argument: 
              Port [type] -> [description]
              
    Parameters: None
    
    Return: None
            
    Edited by: 2020-08-20 Bill Gao
    """    
    cv2.namedWindow('Trackbar')
    cv2.createTrackbar('focus','Trackbar',0,255,lambda tmp:None)
    cap = cv2.VideoCapture(Port,cv2.CAP_DSHOW)

    focus = 0 # min: 0, max: 255, increment:5
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    while (1):
        _, frame = cap.read()
        num_focus = cv2.getTrackbarPos('focus','Trackbar')
        cap.set(cv2.CAP_PROP_FOCUS, num_focus)
        cv2.imshow('frame',frame)


        if cv2.waitKey(1) & 0xFF==ord("q"):
            cv2.destroyAllWindows()  
            break

def findVertex(input_data, row_col = True):
    """
    Function Name: findVertex
    
    Description: Find the vertices of the input_data, return value will be [x, y] instead of [rows, cols]
    
    Argument: 
              input_data [np.array] -> input data
              
    Parameters: None
    
    Return: 
            [np.array] -> coordinate of upper_left
            [np.array] -> coordinate of upper_right
            [np.array] -> coordinate of lower_right
            [np.array] -> coordinate of lower_left
           
    Edited by: 2020-08-20 Bill Gao
    """    
    Left_x  = np.argsort(input_data[:,0])[0:2]
    Right_x = np.argsort(input_data[:,0])[2:4]

    if row_col:
        upper_left  = input_data[Left_x][np.argmin(input_data[Left_x][:,1])]
        upper_right = input_data[Left_x][np.argmax(input_data[Left_x][:,1])]
        lower_left  = input_data[Right_x][np.argmin(input_data[Right_x][:,1])]
        lower_right = input_data[Right_x][np.argmax(input_data[Right_x][:,1])]
    else:
        upper_left  = input_data[Left_x][np.argmin(input_data[Left_x][:,1])]
        upper_right = input_data[Left_x][np.argmax(input_data[Left_x][:,1])]
        lower_left  = input_data[Right_x][np.argmin(input_data[Right_x][:,1])]
        lower_right = input_data[Right_x][np.argmax(input_data[Right_x][:,1])]


    return upper_left, upper_right, lower_right, lower_left