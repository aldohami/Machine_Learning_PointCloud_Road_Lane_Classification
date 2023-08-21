import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Load the image
img = cv.imread('/Classification-of-number-of-lanes-transition-areas-and-crossing-from-point-clouds/img03.jpg')
assert img is not None, "file could not be read, check with os.path.exists()"

# Perform pyramid downsampling
lower_reso = cv.pyrDown(img)

# Display the original and lower resolution images
plt.subplot(121), plt.imshow(img[:, :, ::-1])
plt.title('Original Image'), plt.axis('off')
plt.subplot(122), plt.imshow(lower_reso[:, :, ::-1])
plt.title('Lower Resolution'), plt.axis('off')
plt.show()

higher_reso2 = cv.pyrUp(lower_reso)

plt.subplot(121), plt.imshow(img[:, :, ::-1])
plt.title('Original Image'), plt.axis('off')
plt.subplot(122), plt.imshow(higher_reso2[:, :, ::-1])
plt.title('Lower Resolution'), plt.axis('off')
plt.show()

import cv2 as cv
import numpy as np,sys
A = cv.imread('/Classification-of-number-of-lanes-transition-areas-and-crossing-from-point-clouds/img03.jpg')
B = cv.imread('/Classification-of-number-of-lanes-transition-areas-and-crossing-from-point-clouds/img04.jpg')
assert A is not None, "file could not be read, check with os.path.exists()"
assert B is not None, "file could not be read, check with os.path.exists()"
# generate Gaussian pyramid for A
G = A.copy()
gpA = [G]
for i in range(6):
 G = cv.pyrDown(G)
 gpA.append(G)
# generate Gaussian pyramid for B
G = B.copy()
gpB = [G]
for i in range(6):
 G = cv.pyrDown(G)
 gpB.append(G)
# generate Laplacian Pyramid for A
lpA = [gpA[5]]
for i in range(5,0,-1):
 GE = cv.pyrUp(gpA[i])
 L = cv.subtract(gpA[i-1],GE)
 lpA.append(L)
# generate Laplacian Pyramid for B
lpB = [gpB[5]]
for i in range(5,0,-1):
 GE = cv.pyrUp(gpB[i])
 L = cv.subtract(gpB[i-1],GE)
 lpB.append(L)
# Now add left and right halves of images in each level
LS = []
for la,lb in zip(lpA,lpB):
 rows,cols,dpt = la.shape
 ls = np.hstack((la[:,0:cols//2], lb[:,cols//2:]))
 LS.append(ls)
# now reconstruct
ls_ = LS[0]
for i in range(1,6):
 ls_ = cv.pyrUp(ls_)
 ls_ = cv.add(ls_, LS[i])
# image with direct connecting each half
real = np.hstack((A[:,:cols//2],B[:,cols//2:]))
cv.imwrite('Pyramid_blending2.jpg',ls_)
cv.imwrite('Direct_blending.jpg',real)