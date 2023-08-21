import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import csv

def local_file(file_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file_path = os.path.join(script_dir,'..','dataset', file_name)
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
        '''else:
            col_min = selected_values[:, i].min()
            col_max = selected_values[:, i].max()
            normalized_values[:, i] = (selected_values[:, i] - col_min) / (col_max - col_min)'''
       
    local_x = data[:, 0]
    local_y = data[:, 1]
    x_min, x_max = np.min(local_x), np.max(local_x)
    y_min, y_max = np.min(local_y), np.max(local_y)
    num_cells_x = int(np.ceil((x_max - x_min) / cell_size))
    num_cells_y = int(np.ceil((y_max - y_min) / cell_size))
    size = [num_cells_x,num_cells_y]
    grid = np.zeros((num_cells_y, num_cells_x, 15), dtype=np.float32)
    start_x = x_min
    for num_x in range(num_cells_x):
        start_y = y_min 
        
        for num_y in range(num_cells_y):
            end_x = start_x + cell_size
            end_y = start_y + cell_size 
            
            mask = (data[:, 0] >= start_x) & (data[:, 0] < end_x) & (data[:, 1] >= start_y) & (data[:, 1] < end_y)
            inx = np.where(mask)[0]
            if len(data[inx]) != 0:                           
                grid[num_y, num_x, :] = np.mean(selected_values[inx, :], axis=0)
            start_y = end_y 
            
        start_x = end_x
    
    return size, grid

def save_img(grid, channel_index, file_name, feat, clas):
                
    # TODO change the range with if statment
    image = np.uint8((grid[:, :, channel_index] - grid[:, :, channel_index].min()) * 255 /
                                (grid[:, :, channel_index].max() - grid[:, :, channel_index].min()))
    
    #image = cv2.resize(image, (300, 600), interpolation=cv2.INTER_LINEAR)
    image = cv2.flip(image, 0)
    folder_name = clas + "_" + "img"
    file_base = os.path.splitext(file_name)[0]
    new_file_name = file_base + "_" + feat + ".png"
    file_path = os.path.join(local_file(folder_name), new_file_name)
    cv2.imwrite(file_path, image) 

def save_list_to_csv(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write the header
        writer.writerow(['num_cells_x', 'num_cells_y'])
        
        # Write the data
        writer.writerows(data)


data = np.load("/Users/emadaldoghry/Repo/Classification-of-number-of-lanes-transition-areas-and-crossing-from-point-clouds/dataset/transition/Tile6.npy")

channel_index = 3
size, grid = pc_2_img(data, 0.3)

image_01 = np.uint8(grid[:, :, channel_index]*255)

# INTER_NEAREST, INTER_LINEAR, INTER_AREA, INTER_CUBIC, INTER_LANCZOS4
image_01 = cv2.resize(image_01, (100, 80), interpolation=cv2.INTER_NEAREST)
image_01 = cv2.flip(image_01, 0)

cv2.imwrite('img_norm.jpg', image_01) 

# Scale the normalized grid to the range [0, 255] and convert to uint8
image_02 = np.uint8(grid*255)

#image_02= cv2.resize(image_02, (100, 80), interpolation=cv2.INTER_NEAREST)
image_02 = cv2.flip(image_02, 0)

#np.save('image.npy', image_02)

# Extract the first three channels (assuming RGB format)
rgb_image = image_02[:, :, :3]

# Convert the RGB image to BGR format
bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

# Save the BGR image as JPG
cv2.imwrite('image.jpg', bgr_image)

'''classes = ['2lanes', '3lanes', 'crossing', 'split4lanes', 'split6lanes', 'transition']
features = ['x', 'y', 'z', 'red', 'green', 'blue', 'global_x', 'global_y', 'global_z',
            'intensity', 'red_returns', 'planarity', 'linearity', 'sphericity', 'verticality',
            'mean_int_0.3m_y', 'mean_int_1.5m_y', 'mean_int_0.3m_x', 'mean_int_1.5m_x',
            'edge_area', 'grid_incr_0.3m', 'int_grad_pos']

value_list = []
for clas in classes:
    folder_path = local_file(clas)
    for feature in range(22):
        feat = features[feature]
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                data = np.load(file_path)
                size, grid = pc_2_img(data, 0.3)
                value_list.append(size)
                save_img(grid, feature, file_name, feat, clas)

save_list_to_csv(value_list, 'output.csv')'''