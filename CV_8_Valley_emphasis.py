import cv2
import numpy as np
import os 


image = cv2.imread('./Image/lena.png',0)

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
    print( "manual otsu: {}, opencv otsu: {}".format(thresh,ret) )
    
    return thresh


def valley_emphasis(input_image):
    """
        follow the formula from the thsis:
        Automatic thersholding for defect detection - Hui-Fuang Ng 2006
    """
    blur = cv2.GaussianBlur(input_image,(5,5),0)
    # find normalized_histogram, and its cumulative distribution function
    hist = cv2.calcHist([blur],[0],None,[256],[0,256])
    hist_norm = hist.ravel()/hist.sum()

    # cumulative distribution function (CDF) for probabilities
    omega_total = hist_norm.cumsum()
    bins = np.arange(256)
    fn_max = 0
    valley = 0
    for i in range(256):
        # probabilities
        p1, p2 = np.hsplit(hist_norm,[i]) 
        omega1, omega2 = omega_total[i], omega_total[255]-omega_total[i] # cum sum of classes

        # make sure it would not divide by zero 
        if omega1 < 1.e-6 or omega2 < 1.e-6:
            continue

        # weights
        i1, i2 = np.hsplit(bins,[i])
        
        # finding means and variances
        mu1, mu2 = np.sum(p1*i1)/omega1, np.sum(p2*i2)/omega2

        # calculates the minimization function
        fn_tmp = (1-hist_norm[i])*(omega1*(mu1**2) + omega2*(mu2**2))
        if fn_tmp > fn_max:
            fn_max = fn_tmp
            valley = i

    # otsu
    otsu_value = otsu(input_image)
    print("otsu: {}, valley: {}".format(otsu_value, valley))
    

valley_emphasis(image)
