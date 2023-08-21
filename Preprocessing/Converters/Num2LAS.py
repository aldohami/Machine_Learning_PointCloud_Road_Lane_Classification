' convert the Numpy Files to LAS'

import numpy as np
import laspy
import os

def local_file(file_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file_path = os.path.join(script_dir,'dataset', file_name)
    return data_file_path

'''script_dir = os.path.dirname(os.path.abspath(__file__))
filename="new_file.las"
data_file_path = os.path.join(script_dir, filename)
las = laspy.read(data_file_path)
point_format = las.point_format
print(list(point_format.dimension_names))'''


folder_path = local_file("transition")

# 1. Create a new header
header = laspy.LasHeader(point_format=2, version="1.2")
#header.add_extra_dim(laspy.ExtraBytesParams(name="random", type=np.int32))
#header.offsets = np.min(my_data, axis=0)
#header.scales = np.array([0.1, 0.1, 0.1])

for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    if os.path.isfile(file_path):
        data = np.load(file_path)
        
        las = laspy.LasData(header)
        las.x = data[:, 0]
        las.y = data[:, 1]
        las.z = data[:, 2]
        las.red = data[:, 3]
        las.green = data[:, 4]
        las.blue = data[:, 5]
        las.intensity = data[:, 9]
        las.number_of_returns = data[:, 10]

        folder_path_LAS = local_file("transition_LAS")
        file_base = os.path.splitext(file_name)[0]
        new_file_name = file_base + ".las"
        file_path_LAS = os.path.join(folder_path_LAS, new_file_name)
        las.write(file_path_LAS)








