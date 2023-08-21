import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import csv

def local_file(file_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file_path = os.path.join(script_dir,'dataset', file_name)
    return data_file_path

def pc_2_img(data, cell_size):

    selected_columns = [3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21]
    selected_values = data[:, selected_columns]
    
    for i, col in enumerate(selected_columns):
        if col in (3, 4, 5):
            selected_values[:, i] = selected_values[:, i] / 255.0
        elif col in (9, 15, 16, 17, 18, 21):
            selected_values[:, i] = selected_values[:, i] / 65535.0
        elif col == 10:
            selected_values[:, i] = np.ceil(selected_values[:, i] / 6)
        elif col in (11, 12, 13, 14):
            selected_values[:, i] = selected_values[:, i]
        else:
            selected_values[:, i] = selected_values[:, i] / 1584.0
    #print(np.array(selected_values).shape)
    zeros_array = np.zeros(len(selected_values), dtype=float)
    for i, pnt in enumerate(selected_values):
        zeros_array[i] = (selected_values[i,0] * 100 +
                             selected_values[i,1] * 100 + 
                             selected_values[i,2] * 100 + 
                             selected_values[i,3] * 1000 + 
                             selected_values[i,4] + 
                             selected_values[i,5] + 
                             selected_values[i,6] + 
                             selected_values[i,7] + 
                             selected_values[i,8] + 
                             selected_values[i,9] * 10+ 
                             selected_values[i,10] * 10+ 
                             selected_values[i,11] * 10+ 
                             selected_values[i,12] * 10+ 
                             selected_values[i,13] + 
                             selected_values[i,14])/1347 
    local_x = data[:, 0]
    local_y = data[:, 1]
    x_min, x_max = np.min(local_x), np.max(local_x)
    y_min, y_max = np.min(local_y), np.max(local_y)
    num_cells_x = int(np.ceil((x_max - x_min) / cell_size))
    num_cells_y = int(np.ceil((y_max - y_min) / cell_size))
    size = [num_cells_x,num_cells_y]
    grid = np.zeros((num_cells_y, num_cells_x, 1), dtype=np.float32)
    start_x = x_min
    for num_x in range(num_cells_x):
        start_y = y_min 
        
        for num_y in range(num_cells_y):
            end_x = start_x + cell_size
            end_y = start_y + cell_size 
            
            mask = (data[:, 0] >= start_x) & (data[:, 0] < end_x) & (data[:, 1] >= start_y) & (data[:, 1] < end_y)
            inx = np.where(mask)[0]
            if len(data[inx]) != 0:                           
                grid[num_y, num_x] = np.mean(zeros_array[inx], axis=0)
            start_y = end_y 
            
        start_x = end_x
    
    return size, grid


def save_img(grid, file_name, clas, file_type = "numpy"):
    if file_type == "numpy":
        folder_name = clas + "_" + "npy"
        if not os.path.exists(local_file(folder_name)):
            try:
                # Create the new folder
                os.makedirs(local_file(folder_name))

                print(f"Folder '{folder_name}' created successfully at '{local_file(folder_name)}'.")
            except OSError as e:
                print(f"Error creating folder: {e}")
        
        image_npy = np.float32(grid)
        # Get the original dimensions
        original_height, original_width = image_npy.shape[:2]

        # Calculate the new dimensions
        new_width = original_width * 100
        new_height = original_height * 100
        # INTER_NEAREST, INTER_LINEAR, INTER_AREA, INTER_CUBIC, INTER_LANCZOS4
        resized_image = cv2.resize(input_image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        image_npy = cv2.flip(image_npy, 0)
        
        file_path_npy = os.path.join(local_file(folder_name), file_name) 
        
        np.save(file_path_npy, image_npy)
    
    else:     
        folder_name = clas + "_" + "img"
        if not os.path.exists(local_file(folder_name)):
            try:
                # Create the new folder
                os.makedirs(local_file(folder_name))

                print(f"Folder '{folder_name}' created successfully at '{local_file(folder_name)}'.")
            except OSError as e:
                print(f"Error creating folder: {e}")
        
        
        image_img = np.float32(grid[:, :, :3]*255)
        rgb_image = cv2.cvtColor(image_img, cv2.COLOR_GRAY2RGB)
        file_base = os.path.splitext(file_name)[0]
        new_file_name = file_base + "_" + ".jpg"
        file_path_jpg = os.path.join(local_file(folder_name), new_file_name)  
    
        cv2.imwrite(file_path_jpg, rgb_image) 


def save_list_to_csv(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write the header
        writer.writerow(['num_cells_x', 'num_cells_y'])
        
        # Write the data
        writer.writerows(data)

classes = ['2lanes', '3lanes', 'crossing', 'split4lanes', 'split6lanes', 'transition']
features = ['x', 'y', 'z', 'red', 'green', 'blue', 'global_x', 'global_y', 'global_z',
            'intensity', 'red_returns', 'planarity', 'linearity', 'sphericity', 'verticality',
            'mean_int_0.3m_y', 'mean_int_1.5m_y', 'mean_int_0.3m_x', 'mean_int_1.5m_x',
            'edge_area', 'grid_incr_0.3m', 'int_grad_pos']

#value_list = []
for clas in classes:
    folder_path = local_file(clas)
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            data = np.load(file_path)
            size, grid = pc_2_img(data, 0.3)
            #value_list.append(size)
            save_img(grid, file_name, clas, file_type = "img")

#save_list_to_csv(value_list, 'output.csv')                
                
         
