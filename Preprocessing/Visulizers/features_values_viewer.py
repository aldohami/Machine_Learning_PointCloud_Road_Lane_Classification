'Code to Visulize the Value of the features of our data'

import numpy as np
import os
import matplotlib.pyplot as plt


def local_file(file_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file_path = os.path.join(script_dir,'dataset', file_name)
    return data_file_path

def save_plot(merged_list, feat, clas):
    plt.figure()
    plt.plot(merged_list)
    plt.xlabel('Number of values')  
    plt.ylabel(feat)  
    plt.title(f'Class: {clas}') 
    plt.grid(True)  

    new_file_name = clas + "_" + feat + ".png"
    file_path = os.path.join(local_file('features_values_imgs'), new_file_name)
    plt.savefig(file_path)
        

classes = ['2lanes', '3lanes', 'crossing', 'split4lanes', 'split6lanes', 'transition']
features = ['local x', 'local y', 'local z', 'red values', 'green values', 'blue values',
            'global x', 'global y', 'global z', 'intensity ', 'red number of lidar returns', 'planarity ',
            'linearity ','sphericity ','verticality ', 'mean intensity in 0.3m increments along y', 
            ' mean intensity in 1.5m increments along y', 'mean intensity in 0.3m increments along x',
            'mean intensity in 1.5m increments along x', 'edge area', 'grid increment index 0.3m resolution',
            'intensity principal gradient positions']
print(len(features))

for clas in classes:
    folder_path = local_file(clas)
    for feature in range(22):
        value_list = []
        feat = features[feature]
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                data = np.load(file_path)
                array = data[:,feature]
                value_list.append(array)
         
        merged_array = np.concatenate(value_list)
        merged_list = merged_array.tolist() 
        save_plot(merged_list, feat, clas)