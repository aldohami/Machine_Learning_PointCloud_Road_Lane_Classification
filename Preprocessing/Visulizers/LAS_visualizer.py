' code to show the generated LAS Point Cloud '

import numpy as np
import laspy
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors


def local_file(file_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file_path = os.path.join(script_dir,'dataset', file_name)
    return data_file_path


folder_path = local_file("transition_LAS")

for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    if os.path.isfile(file_path):
        las = laspy.read(file_path)
        
        # Get the x, y, and z coordinates of the points.
        x = las.x
        y = las.y
        z = las.z

        # Get the intensity values of the points.
        intensity = las.intensity

        # Create a scatter plot of the points, coloring them by intensity.
        plt.scatter(x, y, c=intensity, cmap="jet")

        # Add a title to the plot.
        plt.title("LAS Point Cloud")

        # Add a legend to the plot.
        plt.legend(["Intensity"], loc="upper left")

        # Show the plot.
        plt.show()


# Get the X, Y, and Z coordinates from the LAS file
x = las.x
y = las.y
z = las.z

# Get the RGB color values from the LAS file
red = las.red
green = las.green
blue = las.blue

# Normalize the RGB color values to [0, 1]
red = red / 65535.0
green = green / 65535.0
blue = blue / 65535.0

# Create a colormap using the color values
color_values = list(zip(red, green, blue))
cmap = colors.ListedColormap(color_values)

# Plot the point cloud with color mapping
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(x, y, z, c=color_values, cmap=cmap, s=0.1)

# Set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Add a colorbar
cbar = fig.colorbar(scatter)

# Show the plot
plt.show()