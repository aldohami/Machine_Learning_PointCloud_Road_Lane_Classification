import cv2 as cv
import cv2 
import numpy as np
import os
from matplotlib import pyplot as plt

# Update the file path to your image
image_path = '/Classification-of-number-of-lanes-transition-areas-and-crossing-from-point-clouds/img03.jpg'

# Check if the file exists
assert os.path.exists(image_path), "File could not be found, check the path."

# Read the image in grayscale
img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

'''mask = np.where(img == 0, 1, 0).astype(np.uint8)

# Apply morphological dilation to fill gaps
kernel = np.ones((2, 2), np.uint8)
filled_mask = cv2.dilate(img, kernel, iterations=1)

# Inpainting to fill the gaps based on the filled mask
img = cv2.inpaint(img, filled_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)'''

# Check if the image was read successfully
assert img is not None, "File could not be read."

# Perform the Fourier Transform
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20 * np.log(np.abs(fshift))

# Plot the images
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

plt.show()

rows, cols = img.shape
crow,ccol = rows//2 , cols//2
fshift[crow-30:crow+31, ccol-30:ccol+31] = 0
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.real(img_back)
plt.subplot(131),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(img_back, cmap = 'gray')
plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(img_back)
plt.title('Result in JET'), plt.xticks([]), plt.yticks([])
plt.show()

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
#img = cv.imread('messi5.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
dft = cv.dft(np.float32(img),flags = cv.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20*np.log(cv.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Read the image in grayscale
#img = cv.imread('/Users/emadaldoghry/Repo/Classification-of-number-of-lanes-transition-areas-and-crossing-from-point-clouds/img03.jpg', cv.IMREAD_GRAYSCALE)

# Get the shape of the image
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2

# Create a mask first, center square is 1, remaining all zeros
mask = np.zeros((rows, cols, 2), np.uint8)
mask[crow - 30:crow + 30, ccol - 30:ccol + 30, :] = 1

# Apply DFT
dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# Apply mask to the shifted DFT
fshift = dft_shift * mask

# Inverse DFT
f_ishift = np.fft.ifftshift(fshift)
img_back = cv.idft(f_ishift)
img_back = cv.magnitude(img_back[:, :, 0], img_back[:, :, 1])

# Display the images
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_back, cmap='gray')
plt.title('Filtered Image'), plt.xticks([]), plt.yticks([])
plt.show()

