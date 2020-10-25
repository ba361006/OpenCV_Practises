# 2020.10.25
import cv2 
import numpy as np 
from logger import Log
import tools
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
        input_image[:,:,channel] = (copy_image[:,:,channel]/256)*intensity_level
        # print("input_image:", input_image[:,:,channel][0])
        # print("copy_image:", copy_image[:,:,channel][0])
    
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
    plt.show()


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
    if kernel_size%2 == 0 or kernel_size < 3:
        logger.show("Size of kernel should be odd and be greater or equal to 3", "error")
        quit()
    
    kernel = np.ones([kernel_size, kernel_size])/ kernel_size
    kernel = np.float(kernel)
    print(kernel)
    # lena = tools.slidingWindow(input_image, kernel)
    # cv2.imshow("orginal_image", input_image)
    # cv2.imshow("lena", lena)
    # cv2.waitKey()
    # if style.upper() == "SAME":
    
    # if style.upper() == "SMALLER"

    

if __name__ == "__main__":
    logger = Log("Main", stream_level="debug")
    lena = cv2.imread("./Image/lena.png")
    # task_1(input_image=lena, intensity_level=64)
    task_2(input_image=lena, kernel_size=5, style = "same")

