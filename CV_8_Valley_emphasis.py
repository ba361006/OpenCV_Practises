import cv2
import numpy as np
import os 
import tools



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

    # Omega_total cumulative distribution function (CDF)
    Q = hist_norm.cumsum()
    
    bins = np.arange(256)
    fn_min = np.inf
    thresh = -1
    for i in range(256):
        # probabilities
        p1, p2 = np.hsplit(hist_norm,[i]) 

        # cum sum of classes
        q1, q2 = Q[i],Q[255]-Q[i] 

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

def v_otsu(input_image):
    """
        follow the formula from https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html
    """
    blur = cv2.GaussianBlur(input_image,(5,5),0)
    # find normalized_histogram, and its cumulative distribution function
    hist = cv2.calcHist([blur],[0],None,[256],[0,256])
    hist_norm = hist.ravel()/hist.sum()

    # cumulative distribution function (CDF)
    omega_total = hist_norm.cumsum()
    
    bins = np.arange(256)
    fn_min = np.inf
    thresh = -1
    for i in range(256):
        # probabilities
        p1, p2 = np.hsplit(hist_norm,[i]) 
        omega1, omega2 = omega_total[i], omega_total[255]- omega_total[i] # cum sum of classes

        # make sure it would not divide by zero 
        if omega1 < 1.e-6 or omega2 < 1.e-6:
            continue

        # weights
        i1, i2 = np.hsplit(bins,[i])
        
        # finding means and variances
        mu1, mu2 = np.sum(i1*p1)/omega1, np.sum(i2*p2)/omega2
        var1, var2 = np.sum(((i1-mu1)**2)*p1)/omega1, np.sum(((i2-mu2)**2)*p2)/omega2

        # calculates the minimization function
        fn_tmp = var1*omega1 + var2*omega2
        # fn_tmp = omega1 * var1^2 + omega2*var2^2
        if fn_tmp < fn_min:
            fn_min = fn_tmp
            thresh = i

    # find otsu's threshold value with Opencv2 function
    ret, otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # print( "manual otsu: {}, opencv otsu: {}".format(thresh,ret) )
    
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
    fn_min = np.inf
    valley = -1
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
        mu1, mu2 = np.sum(i1*p1)/omega1, np.sum(i2*p2)/omega2
        var1, var2 = np.sum(((i1-mu1)**2)*p1)/omega1, np.sum(((i2-mu2)**2)*p2)/omega2

        # calculates the minimization function
        # fn_tmp = (1-hist_norm[i])*(omega1*(mu1**2) + omega2*(mu2**2))
        fn_tmp = (1-hist_norm[i])*(omega1*var1 + omega2*var2)
        if fn_tmp < fn_min:
            fn_min = fn_tmp
            valley = i

    # otsu
    otsu_value = otsu(input_image)
    # print("otsu: {}, valley: {}".format(otsu_value, valley))

    return valley

# for idx in range(18):
#     name = "./Image/original_sample_" + str(idx) + ".jpg"
#     image = cv2.imread(name,0)
#     thresh_valley = valley_emphasis(image)
#     thresh_otsu = v_otsu(image)
#     print("\n================ Final Threshold value ================")
#     print("\nOtus: {}, valley: {}".format(thresh_otsu, thresh_valley))
#     otsu_bin, otsu_colour = tools.threshold(image,0,thresh_otsu)
#     valley_bin, valley_colour = tools.threshold(image,0,thresh_valley)
#     cv2.imshow("otsu_bin", otsu_bin)
#     cv2.imshow("valley_bin", valley_bin)
#     cv2.waitKey()
