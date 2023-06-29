import numpy as np
import cv2

def compute_hist(img):
    hist = [0] * 256
    hist = np.array(hist)
    (h,w) = img.shape[:2]
    for i in range(h):
        for j in range(w):
            light_intensity = img[i][j]
            hist[light_intensity] += 1
    return hist

def equalizeHist(hist):
    cum = np.zeros_like(hist)
    for i in range(len(cum)):
        cum[i] = hist[:i].sum()
    # scale cum's elements:
    cum = (cum - cum.min())/(cum.max() - cum.min()) * 255
    cum = [int(i) for i in cum]
    cum = np.array(cum)
    return cum

def brightness_colored(img):
    hist = compute_hist(img)
    cum = equalizeHist(hist)
    new_img = np.zeros_like(img)
    (h, w) = new_img.shape[:2]
    for i in range(h):
        for j in range(w):
            new_img[i][j] = cum[img[i][j]]
    return new_img

def brightness_colord_by_cv2(img):
    # split image into separate color channels
    b, g, r = cv2.split(img)

    # perform histogram equalization on each channel
    b_eq = cv2.equalizeHist(b)
    g_eq = cv2.equalizeHist(g)
    r_eq = cv2.equalizeHist(r)

    # merge equalized channels back into a color image
    equ = cv2.merge((b_eq, g_eq, r_eq))

    '''# display original and equalized images
    cv2.imshow('Original Image', img)
    cv2.imshow('Equalized Image', equ)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''

    return equ

def medianBlur_by_cv2(img):
    # apply median filter with kernel size 5x5
    median = cv2.medianBlur(img, 5)

    '''# display original and median-filtered images
    cv2.imshow('Original Image', img)
    cv2.imshow('Median-Filtered Image', median)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''

    return median