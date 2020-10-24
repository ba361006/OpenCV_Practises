import cv2
import numpy as np
import os 
import tools
import time
from logger import Log

def otsu(input_image):
    """
        follow the formula from https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html
    """
    blur = cv2.GaussianBlur(input_image,(5,5),0)
    # find normalized_histogram, and its cumulative distribution function
    hist = cv2.calcHist([blur],[0],None,[256],[0,256])
    hist_norm = hist.ravel()/hist.sum()

    # cumulative distribution function (CDF)
    Q = hist_norm.cumsum()
    
    bins = np.arange(256)
    fn_min = np.inf
    thresh = -1
    for i in range(256):
        # probabilities
        p1, p2 = np.hsplit(hist_norm,[i]) 
        q1, q2 = Q[i],Q[255]-Q[i] # cum sum of classes

        # make sure it would not divide by zero 
        if q1 < 1.e-6 or q2 < 1.e-6:
            continue

        # weights
        i1, i2 = np.hsplit(bins,[i])
        
        # finding means and variances
        mu1, mu2 = np.sum(p1*i1)/q1, np.sum(p2*i2)/q2
        var1, var2 = np.sum(((i1-mu1)**2)*p1)/q1, np.sum(((i2-mu2)**2)*p2)/q2

        # calculates the minimization function
        fn_tmp = var1*q1 + var2*q2
        if fn_tmp < fn_min:
            fn_min = fn_tmp
            thresh = i

    # find otsu's threshold value with Opencv2 function
    ret, otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # print( "manual otsu: {}, opencv otsu: {}".format(thresh,ret) )
    
    return thresh

def otsu_revised(input_image):
    """
        follow the formula from https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html
    """
    blur = cv2.GaussianBlur(input_image,(5,5),0)
    # find normalized_histogram, and its cumulative distribution function
    hist = cv2.calcHist([blur],[0],None,[256],[0,256])
    hist_norm = hist.ravel()/hist.sum()

    # omega_total cumulative distribution function (CDF)
    omega_total = hist_norm.cumsum()

    bins = np.arange(256)
    fn_min = np.inf
    thresh = -1
    for i in range(256):
        # probabilities, np.hsplit will separate variable by the specified number
        # ex: a, b = np.hsplit(variable, 1) -> a=variable[0], b=variable[1~L-1], L = len(variable)
        # but if np.hsplit(variable, 0) -> a=0, b=variable[0~L-1], L = len(variable)
        p1, p2 = np.hsplit(hist_norm,[i+1]) 

        # get omega_1 and omega_2
        omega_1, omega_2 = omega_total[i], omega_total[255]-omega_total[i] 

        # make sure it would not divide by zero 
        if omega_1 < 1.e-6 or omega_2 < 1.e-6:
            # omega_1 = 1.e-6
            # omega_2 = 1.e-6
            continue

        # weights
        i1, i2 = np.hsplit(bins,[i+1])
        
        # computing mu1, mu2    
        mu_total = (hist_norm * bins)[i]
        mu1 = np.sum(p1*i1)/omega_1
        mu2 = np.sum(p2*i2)/omega_2
        # var1, var2 = np.sum(((i1-mu1)**2)*p1/omega_1, np.sum(((i2-mu2)**2)*p2)/omega_2

        # calculates the minimization function
        # fn_tmp = var1*omega_1 + var2*omega_2
        t_star = np.sum(omega_1*np.square(mu1-mu_total) + omega_2*np.square(mu2-mu_total))
        if t_star < fn_min:
            fn_min = t_star
            thresh = i

        # logger.show("step: {}".format(i), "debug")
        # logger.show("p1: {}, p2: {}".format(p1, p2), "debug")
        # logger.show("omega_1: {}, omega_2: {}".format(omega_1, omega_2), "debug")
        # logger.show("mu1: {}, m2: {}".format(mu1, mu2), "debug")
        # logger.show("t_star: {}".format(t_star), "debug")

        # os.system("pause")
    # find otsu's threshold value with Opencv2 function
    ret, otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    print( "manual otsu: {}, opencv otsu: {}".format(thresh,ret) )
    
    return thresh

logger = Log("Main", "info")
for idx in range(10):
    name = "./Image/multi_thresh_" + str(idx) + ".tif"
    multi = cv2.imread(name, 0)
    # # multi = cv2.resize(multi, (int(multi.shape[1]/3),int(multi.shape[0]/3)))
    valley = otsu_revised(multi)
    otsu1 = otsu(multi)
    valley_bin, valley_colour = tools.threshold(multi,0,valley)
    otsu_bin, otsu_colour = tools.threshold(multi, 0, otsu1)

    cv2.imshow("valley_bin", valley_bin)
    cv2.imshow("otsu_bin", otsu_bin)
    cv2.imshow("multi", multi)
    cv2.waitKey()

# for idx in range(18):
#     name = "./Image/original_sample_" + str(idx) + ".jpg"
#     image = cv2.imread(name,0)
#     thresh_valley = otsu_revised(image)
#     thresh_otsu = otsu(image)
#     print("\n================ Final Threshold value ================")
#     print("\nOtus: {}, valley: {}".format(thresh_otsu, thresh_valley))
#     otsu_bin, otsu_colour = tools.threshold(image,0,thresh_otsu)
#     valley_bin, valley_colour = tools.threshold(image,0,thresh_valley)
#     cv2.imshow("otsu_bin", otsu_bin)
#     cv2.imshow("valley_bin", valley_bin)
#     cv2.waitKey()

