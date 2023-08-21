import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('/Classification-of-number-of-lanes-transition-areas-and-crossing-from-point-clouds/img03.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
kernel = np.ones((5,5),np.uint8)
erosion = cv.erode(img,kernel,iterations = 1)

combined = np.hstack((img, erosion))
cv.imshow('Original vs Erosion', combined)
cv.waitKey(0)
cv.destroyAllWindows()

dilation = cv.dilate(img,kernel,iterations = 1)
cv.imshow('Original vs Erosion', dilation)
cv.waitKey(0)
cv.destroyAllWindows()

opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
cv.imshow('Original vs Erosion', dilation)
cv.waitKey(0)
cv.destroyAllWindows()

closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
cv.imshow('Original vs Erosion', dilation)
cv.waitKey(0)
cv.destroyAllWindows()

gradient = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)
cv.imshow('Original vs Erosion', gradient)
cv.waitKey(0)
cv.destroyAllWindows()

tophat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)
cv.imshow('Original vs Erosion', tophat)
cv.waitKey(0)
cv.destroyAllWindows()

blackhat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)
cv.imshow('Original vs Erosion', blackhat)
cv.waitKey(0)
cv.destroyAllWindows()
