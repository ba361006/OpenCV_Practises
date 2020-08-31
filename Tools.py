import numpy as np 
import cv2
import sys
import traceback

def nothing(x):
    pass


def Threshold_test(Image_in, BGR = False):
    if len(Image_in.shape) == 3:
        if BGR == True:
            Image_in = cv2.cvtColor(Image_in, cv2.COLOR_BGR2GRAY)
        else:
            Image_in = cv2.cvtColor(Image_in, cv2.COLOR_RGB2GRAY)

    cv2.namedWindow("Trackbars")
    cv2.createTrackbar("High", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("Low", "Trackbars", 0, 255, nothing)
    while (True):
        High = cv2.getTrackbarPos("High", "Trackbars")
        Low = cv2.getTrackbarPos("Low", "Trackbars")
        Binary_img = cv2.inRange(Image_in, Low, High )
        Bitwise_img = cv2.bitwise_and(Image_in, Image_in, mask = Binary_img)
        cv2.imshow("Threshold_test_Out_Binary", Binary_img)
        cv2.imshow("Threshold_test_Out_Colour", Bitwise_img)

        if cv2.waitKey(1) & 0xFF == ord("w"):
            print("Low, High = {}, {}\n".format(Low, High))
            break
    cv2.destroyWindow("Trackbars")
    cv2.destroyWindow("Threshold_test_Out_Binary")
    cv2.destroyWindow("Threshold_test_Out_Colour")



def Threshold(Image_in, Low, High, Show_Image = None, BGR = False):
    if len(Image_in.shape) == 3:
        if BGR == True:
            Image_in = cv2.cvtColor(Image_in, cv2.COLOR_BGR2GRAY)
        else:
            Image_in = cv2.cvtColor(Image_in, cv2.COLOR_RGB2GRAY)

    Binary_img = cv2.inRange(Image_in, Low, High)
    Bitwise_img = cv2.bitwise_and(Image_in, Image_in, mask = Binary_img)
    
    if Show_Image:
        cv2.imshow("Threshold_Out_Binary", Binary_img)
        cv2.imshow("Threshold_Out_Colour", Bitwise_img)
        cv2.waitKey(0)
        cv2.destroyWindow("Threshold_Out_Binary")
        cv2.destroyWindow("Threshold_Out_Colour")
    return Binary_img, Bitwise_img



def Hsv_test(Image_in, BGR = False):
    if BGR == True:
        img_hsv = cv2.cvtColor(Image_in, cv2.COLOR_BGR2HSV)
    else:
        img_hsv = cv2.cvtColor(Image_in, cv2.COLOR_RGB2HSV)

    cv2.namedWindow("Trackbars")
    cv2.createTrackbar("High_H", "Trackbars", 180, 180, nothing)
    cv2.createTrackbar("High_S", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("High_V", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("Low_H", "Trackbars", 0, 180, nothing)
    cv2.createTrackbar("Low_S", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("Low_V", "Trackbars", 0, 255, nothing)
    print("Press 'w' to end the while loop")
    while (True):
        High_H = cv2.getTrackbarPos("High_H", "Trackbars")
        High_S = cv2.getTrackbarPos("High_S", "Trackbars")
        High_V = cv2.getTrackbarPos("High_V", "Trackbars")
        Low_H = cv2.getTrackbarPos("Low_H", "Trackbars")
        Low_S = cv2.getTrackbarPos("Low_S", "Trackbars")
        Low_V = cv2.getTrackbarPos("Low_V", "Trackbars")

        Low = np.array([Low_H, Low_S, Low_V], dtype = np.uint8)
        High = np.array([High_H, High_S, High_V], dtype = np.uint8) 
        Binary_img = cv2.inRange(img_hsv,Low ,High )
        Bitwise_img = cv2.bitwise_and(Image_in, Image_in, mask = Binary_img)
        cv2.imshow("Hsv_test_threshold", Binary_img)
        cv2.imshow("Hsv_test_bitwise", Bitwise_img)

        if cv2.waitKey(1) & 0xFF == ord("w"):
            print("Low, High = [{},{},{}], [{},{},{}]\n".format(Low[0], Low[1], Low[2], High[0], High[1], High[2]))
            break
    cv2.destroyWindow("Hsv_test_Out")
    cv2.destroyWindow("Hsc_test_biwsie")
    cv2.destroyWindow("Trackbars")



def Hsv(Image_in, Low, High, Show_Image = None, BGR = False):
    if BGR == True:
        img_hsv = cv2.cvtColor(Image_in, cv2.COLOR_BGR2HSV)
        Low = np.array(Low, dtype = np.uint8)
        High = np.array(High, dtype = np.uint8)
    else:
        img_hsv = cv2.cvtColor(Image_in, cv2.COLOR_RGB2HSV)
        Low = np.array(Low, dtype = np.uint8)
        High = np.array(High, dtype = np.uint8)

    
    Binary_img = cv2.inRange(img_hsv, Low, High)
    Bitwise_img = cv2.bitwise_and(Image_in, Image_in, mask = Binary_img)

    if Show_Image:
        cv2.imshow("Hsv_bin", Binary_img)
        cv2.imshow("Hsv_Colour", Bitwise_img)
        cv2.waitKey(0)
        cv2.destroyWindow("Hsv_bin")
        cv2.destroyWindow("Hsv_Colour")
    return Binary_img, Bitwise_img
        

def Canny_test(Image_in, BGR = False):
    # if BGR == True:
    #     Image_in = cv2.cvtColor(Image_in, cv2.COLOR_BGR2GRAY)
    # else:
    #     Image_in = cv2.cvtColor(Image_in, cv2.COLOR_RGB2GRAY)


    blurred = cv2.GaussianBlur(Image_in, (3,3), 0)
    cv2.namedWindow("Trackbars")
    cv2.createTrackbar("High", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("Low", "Trackbars", 0, 255, nothing)
    print("\n Press 'w' to end the while loop")
    while (True):
        High = cv2.getTrackbarPos("High", "Trackbars")
        Low = cv2.getTrackbarPos("Low", "Trackbars")
        canny = cv2.Canny(Image_in , Low, High)
        cv2.imshow("Canny_test_Out", canny)

        if cv2.waitKey(1) & 0xFF == ord("w"):
            print("[Low, High] = [{},{}]".format(Low, High))
            break
    cv2.destroyWindow("Canny_test_Out")


def Canny(Image_in, BGR = False):
    if BGR == True:
        Image_in = cv2.cvtColor(Image_in, cv2.COLOR_BGR2GRAY)
    else:
        Image_in = cv2.cvtColor(Image_in, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(Image_in, (3,3), 0)
    canny = cv2.Canny(blurred,0, 255)
    return canny


# Image_extracting Parameters
mouse_row_0, mouse_col_0, mouse_row_1, mouse_col_1, mouse_count = 0, 0, 0, 0, 0

def mouseaction(event, x, y, flags, param):
    global mouse_row_0, mouse_col_0, mouse_row_1, mouse_col_1, mouse_count
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Left button down")
        if mouse_count == 0:
            mouse_col_0, mouse_row_0 = x, y
            mouse_count = 1
        else:
            mouse_col_1, mouse_row_1 = x, y
            mouse_count = 0
        
def Image_extracting(Image_in, return_image = True):
    global mouse_row_0, mouse_col_0, mouse_row_1, mouse_col_1, mouse_count
    
    cv2.namedWindow("Image_extracting")

    cv2.imshow("Image_extracting", Image_in)
    cv2.setMouseCallback("Image_extracting", mouseaction)
    cv2.waitKey(0)
    
    extract_img = Image_in[mouse_row_0: mouse_row_1, mouse_col_0:mouse_col_1]
    cv2.imshow("extract", extract_img)
    cv2.waitKey(0)
    cv2.destroyWindow("extract")

    if return_image:
        return extract_img



def Sliding_Window(Image_in, kernel):
    img_row, img_col, img_channel = Image_in.shape
    k_row, k_col = kernel.shape

    feature_row = img_row - k_row + 1
    feature_col = img_col - k_col + 1
    feature = np.lib.stride_tricks.as_strided(Image_in,
                                              shape = (1, 
                                                       img_channel, 
                                                       feature_row, 
                                                       feature_col, 
                                                       k_row, 
                                                       k_col),
                                              strides = (1, 
                                                         1, 
                                                         img_channel*img_col, 
                                                         img_channel*1, 
                                                         img_channel*img_col, 
                                                         img_channel*1))

    feature = feature.reshape(feature_row*feature_col*3, k_row, k_col)
    feature_map = np.uint8(np.tensordot(kernel, feature, [(0,1), (1,2)]))

    Processed_image = np.lib.stride_tricks.as_strided(feature_map,
                                                      shape = (feature_row,
                                                               feature_col,
                                                               img_channel),
                                                      strides = (feature_col,
                                                                 1,
                                                                 feature_row * feature_col))

    return Processed_image

def Focus_test(Port):
    cv2.namedWindow('Trackbar')
    cv2.createTrackbar('focus','Trackbar',0,255,nothing)
    cap= cv2.VideoCapture(Port,cv2.CAP_DSHOW)

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

def Find_Vertex(data_in):
    
    Left_x = np.argsort(data_in[:,0])[0:2]
    Right_x = np.argsort(data_in[:,0])[2:4]

    Upper_Left = data_in[Left_x][np.argmin(data_in[Left_x][:,1])]
    Upper_Right = data_in[Left_x][np.argmax(data_in[Left_x][:,1])]
    Lower_Left = data_in[Right_x][np.argmin(data_in[Right_x][:,1])]
    Lower_Right = data_in[Right_x][np.argmax(data_in[Right_x][:,1])]

    
    return Upper_Left, Upper_Right, Lower_Right, Lower_Left


def Error_message(Error):
    error_class = Error.__class__.__name__ #取得錯誤類型
    detail = Error.args[0] #取得詳細內容
    cl, exc, tb = sys.exc_info() #取得Call Stack
    lastCallStack = traceback.extract_tb(tb)[-1] #取得Call Stack的最後一筆資料
    fileName = lastCallStack[0] #取得發生的檔案名稱
    lineNum = lastCallStack[1] #取得發生的行號
    funcName = lastCallStack[2] #取得發生的函數名稱
    errMsg = "File \"{}\", line {}, in {}: [{}] {}".format(fileName, lineNum, funcName, error_class, detail)
    print("\n======================== Error Message ========================")
    print(errMsg)
    print("======================== Error Message ========================\n")