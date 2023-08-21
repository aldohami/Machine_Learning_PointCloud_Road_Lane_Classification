import os
import cv2

# Define the root directory containing your image folders
root_directory = '/Users/emadaldoghry/Repo/Classification-of-number-of-lanes-transition-areas-and-crossing-from-point-clouds/dataset/images'

# Function to apply thresholding to an image
def apply_threshold(image, lower_thresh, upper_thresh):
    _, thresholded = cv2.threshold(image, lower_thresh, upper_thresh, cv2.THRESH_BINARY)
    return thresholded

# Loop through all subdirectories and images
for dirpath, dirnames, filenames in os.walk(root_directory):
    for filename in filenames:
        if filename.endswith(('.jpg', '.png', '.jpeg')):  # Adjust the file extensions as needed
            image_path = os.path.join(dirpath, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            if image is not None:
                # Apply thresholding
                lower_threshold = 135
                upper_threshold = 165
                thresholded_image = apply_threshold(image, lower_threshold, upper_threshold)
                
                # Save the thresholded image in the same directory
                save_path = os.path.join(dirpath, f'thresholded_{filename}')
                cv2.imwrite(save_path, thresholded_image)
                print(f"Thresholded image saved: {save_path}")
            else:
                print(f"Unable to read image: {image_path}")
