import cv2 
import time
import tools
import os
import numpy as np 
from logger import Log
from matplotlib import pyplot as plt



def task_1(input_image:np.ndarray, intensity_level:int):
    """
    Function Name: task_1
    *
    Description: task 1 in week 1
    *
    Argument: None
    *
    Parameters: 
                input_image [np.ndarray] -> input image
                intensity_level [int] -> intensity level 
    *
    Return: None
    *
    Edited by: [2020-10-25] [Bill Gao]
    """    
    # input intensity_level checking
    if intensity_level%2 != 0:
        logger.show("input intensity_level should be divisible by 2", "error")
        quit()
    
    copy_image = input_image.copy()
    for channel in range(input_image.shape[2]):
        input_image[:,:,channel] = np.floor(copy_image[:,:,channel] / (256/intensity_level))
    
    ori_hist = cv2.calcHist([copy_image], [0], None, [256], [0,256])
    proc_hist = cv2.calcHist([input_image], [0], None, [256], [0,256])

    # original histogram
    plt.figure("task_1", figsize=(12,5))
    plt.subplot(121)
    plt.plot(ori_hist)
    plt.axvline(intensity_level, plt.ylim()[0], plt.ylim()[1], color="r", linewidth=1, label="intensity")
    plt.xlabel("intensitive level"), plt.ylabel("number of pixels"), plt.title("original image")

    # processed histogram
    plt.subplot(122)
    plt.plot(proc_hist)
    plt.axvline(intensity_level, plt.ylim()[0], plt.ylim()[1], color="r", linewidth=1, label="intensity")
    plt.xlabel("intensitive level"), plt.ylabel("number of pixels"), plt.title("processed image")

    cv2.imshow("original_image", copy_image)
    cv2.imshow("processed_image", input_image)
    cv2.waitKey()
    plt.show()

    cv2.destroyAllWindows()
    plt.close()


def task_2(input_image:np.ndarray, kernel_size:int, style:str):
    """
    Function Name: task_2
    *
    Description: task 2 in week 1
    *
    Argument: None
    *
    Parameters: 
                input_image [np.ndarray] -> input image
                kernel_size [int] -> size of kernel
                style [str] -> should be one of [same, samller]
    *
    Return: 
    *
    Edited by: [2020-10-25] [Bill Gao]
    """    
    if kernel_size%2 == 0 or kernel_size < 3 and kernel_size.__class__ == int:
        logger.show("Size of kernel should be odd and be greater or equal to 3", "error")
        quit()

    if style.upper() == "SAME":
        kernel = np.ones([kernel_size, kernel_size])/kernel_size**2
        row_image, col_image, channel_image = input_image.shape

        deviation = kernel_size-1
        new_image = np.zeros([row_image + deviation, col_image + deviation, channel_image], dtype=np.uint8)
        new_image[int(deviation/2):row_image+int(deviation/2), int(deviation/2):col_image+int(deviation/2), :] = input_image
        row_image, col_image, channel_image = new_image.shape

        k_row, k_col = kernel.shape
        feature_row = row_image - k_row + 1
        feature_col = col_image - k_col + 1
        feature = np.lib.stride_tricks.as_strided(new_image,
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

        processed_image = np.lib.stride_tricks.as_strided(feature_map,
                                                        shape = (feature_row,
                                                                feature_col,
                                                                channel_image),
                                                        strides = (feature_col,
                                                                    1,
                                                                    feature_row * feature_col))
        print("Shape of input_image: ", input_image.shape)
        print("Shape of processed_image: ", processed_image.shape)
        cv2.imshow("orginal_image", input_image)
        cv2.imshow("processed_image", processed_image)
        cv2.waitKey()
        

    if style.upper() == "SMALLER":
        kernel = np.ones([kernel_size, kernel_size])/kernel_size**2

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

        processed_image = np.lib.stride_tricks.as_strided(feature_map,
                                                        shape = (feature_row,
                                                                feature_col,
                                                                channel_image),
                                                        strides = (feature_col,
                                                                    1,
                                                                    feature_row * feature_col))
        cv2.imshow("orginal_image", input_image)
        cv2.imshow("processed_image", processed_image)
        print("Shape of input_image: ", input_image.shape)
        print("Shape of processed_image: ", processed_image.shape)
        cv2.waitKey()
    cv2.destroyAllWindows()


def task_3(input_image, angle):
    if angle.__class__ != int:
        logger.show("Type of angle should be integer", "error")
        quit()

    row, col, channel = input_image.shape
    centre = (int(col/2), int(row/2))
    matrix = cv2.getRotationMatrix2D(centre, angle, 1)
    processed_image = cv2.warpAffine(input_image, matrix, (col, row))
    cv2.imshow("orginal_image", input_image)
    cv2.imshow("processed_image", processed_image)
    cv2.waitKey()
    cv2.destroyAllWindows()


def task_4(input_image:np.ndarray, kernel_size:int):

    if kernel_size%2 == 0 or kernel_size < 3 and kernel_size.__class__ == int:
        logger.show("Size of kernel should be odd and be greater or equal to 3", "error")
        quit()

    row_image, col_image, channel_image = input_image.shape
    kernel = np.ones([kernel_size, kernel_size])/kernel_size**2

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
    feature_map = np.zeros(feature_row*feature_col*3, dtype=np.uint8)
    
    for idx, matrix in enumerate(feature):
        feature_map[idx] = int(matrix.mean())

    processed_image = np.lib.stride_tricks.as_strided(feature_map,
                                                    shape = (feature_row,
                                                            feature_col,
                                                            channel_image),
                                                    strides = (feature_col,
                                                                1,
                                                                feature_row * feature_col))
    cv2.imshow("orginal_image", input_image)
    cv2.imshow("processed_image", processed_image)
    cv2.waitKey()
    cv2.destroyAllWindows()


    

if __name__ == "__main__":
    logger = Log("Main", stream_level="debug")
    lena = cv2.imread("./Image/lena.png")
    task_1(input_image=lena, intensity_level=64)
    task_2(input_image=lena, kernel_size=9, style = "same")
    task_3(input_image=lena, angle=45)
    task_4(input_image=lena, kernel_size=3)