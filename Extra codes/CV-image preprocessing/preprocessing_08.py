import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Read the image
img = cv.imread('/Classification-of-number-of-lanes-transition-areas-and-crossing-from-point-clouds/img03.jpg')

# Check if the image was read successfully
assert img is not None, "File could not be read, check with os.path.exists()"

# Convert the image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Apply Otsu's thresholding
ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

# Display the original image, grayscale image, and thresholded image
plt.subplot(131), plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(132), plt.imshow(gray, cmap='gray')
plt.title('Grayscale Image'), plt.xticks([]), plt.yticks([])

plt.subplot(133), plt.imshow(thresh, cmap='gray')
plt.title('Thresholded Image'), plt.xticks([]), plt.yticks([])

plt.show()

# noise removal
kernel = np.ones((3, 3), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

# sure background area
sure_bg = cv.dilate(opening, kernel, iterations=3)

# Finding sure foreground area
dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg, sure_fg)

# Display the images
plt.subplot(231), plt.imshow(thresh, cmap='gray')
plt.title('Thresholded Image'), plt.xticks([]), plt.yticks([])

plt.subplot(232), plt.imshow(opening, cmap='gray')
plt.title('Morphological Opening'), plt.xticks([]), plt.yticks([])

plt.subplot(233), plt.imshow(sure_bg, cmap='gray')
plt.title('Sure Background Area'), plt.xticks([]), plt.yticks([])

plt.subplot(234), plt.imshow(dist_transform, cmap='gray')
plt.title('Distance Transform'), plt.xticks([]), plt.yticks([])

plt.subplot(235), plt.imshow(sure_fg, cmap='gray')
plt.title('Sure Foreground Area'), plt.xticks([]), plt.yticks([])

plt.subplot(236), plt.imshow(unknown, cmap='gray')
plt.title('Unknown Region'), plt.xticks([]), plt.yticks([])

plt.show()

# Marker labelling
ret, markers = cv.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers + 1

# Now, mark the region of unknown with zero
markers[unknown == 255] = 0

# Display the markers
plt.imshow(markers, cmap='jet')
plt.title('Markers'), plt.xticks([]), plt.yticks([])
plt.colorbar()
plt.show()

# Perform watershed algorithm
markers = cv.watershed(img, markers)

# Mark the watershed boundary with blue color
img[markers == -1] = [255, 0, 0]

# Display the watershed result
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title('Watershed Result'), plt.xticks([]), plt.yticks([])
plt.show()


