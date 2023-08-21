import numpy as np
import laspy
import matplotlib.pyplot as plt
from skimage.morphology import medial_axis, skeletonize
from skimage.util import invert
import os


def local_file(file_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file_path = os.path.join(script_dir,'dataset', file_name)
    return data_file_path

data_file_path = local_file("2lanes/Tile23.npy")

data = np.load(data_file_path)

header = laspy.header.Header()
x=data[:,0]
y=data[:,1]
z=data[:,2]
intensity=data[:,3]
return_num=data[:,4]
classification=data[:,5]

#las = laspy.read(data_file_path)

'''point_cloud = np.array(
    np.transpose(np.vstack(
        (las.x, las.y, las.z, las.red, las.green, las.blue, las.intensity,
         las.number_of_returns))))'''

point_cloud = np.array(
    np.transpose(np.vstack(
        (data[0], data[1], data[2], data[3], data[4], data[5], data[9],
         data[10]))))
"""
point_cloud_mask = np.where((point_cloud[:,0]>=500) & (point_cloud[:,0]<800) )
point_cloud= point_cloud[point_cloud_mask]
point_cloud_mask_1=  np.where(point_cloud[:,1]>=500)
point_cloud= point_cloud[point_cloud_mask_1]
"""
grid_size =10000
cell_size = 0.1
grid = np.zeros((grid_size, grid_size), dtype=bool)


for point in point_cloud:
    x, y = point[0], point[1]

    cell_x = int(x / cell_size)
    cell_y = int(y / cell_size)
    grid[cell_y, cell_x] = True


skeleton = skeletonize(grid, method='lee')
#np.savetxt("skel.txt",skeleton, delimiter=",", fmt="%s")
print(skeleton.shape)
# display results

skel, distance = medial_axis(grid, return_distance=True)

# Distance to the background for pixels of the skeleton
dist_on_skel = distance * skel

centerline= []
for i in range(skeleton.shape[0]):
    for j in range(skeleton.shape[1]):
        if skeleton[i, j]==True:
            centerline.append([j, i,0])
centerline = np.array(centerline)*0.1

print(centerline.shape)
np.savetxt("centerline.txt", centerline, delimiter=",", fmt="%s")
np.savetxt("cloud.txt", point_cloud, delimiter=",", fmt="%s")
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 4),
                         sharex=True, sharey=True)

ax = axes.ravel()

ax[0].imshow(grid, cmap=plt.cm.gray)
ax[0].axis('off')
ax[0].set_title('original', fontsize=20)

ax[1].imshow(skeleton, cmap=plt.cm.gray)
ax[1].axis('off')
ax[1].set_title('skeleton', fontsize=20)

ax[2].imshow(dist_on_skel, cmap='magma')
ax[2].contour(grid, [0.5], colors='w')
ax[2].set_title('medial_axis')
ax[2].axis('off')
fig.tight_layout()
plt.show()

