import cv2 as cv
import numpy as np
from tkinter import filedialog
import tkinter as tk
from PIL import Image, ImageTk

def select_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        process_image(file_path)

def process_image(image_path):
    frame = cv.imread(image_path)

    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Define range of white color in HSV
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([255, 30, 255])

    # Threshold the HSV image to get only white colors
    mask = cv.inRange(hsv, lower_white, upper_white)

    # Bitwise-AND mask and original image
    res = cv.bitwise_and(frame, frame, mask=mask)

    # Display the processed image
    cv.imshow('Original Image', frame)
    cv.imshow('Mask', mask)
    cv.imshow('Processed Image', res)
    cv.waitKey(0)
    cv.destroyAllWindows()

# Create a basic GUI using Tkinter to select images
root = tk.Tk()
root.title("Image Color Masking")

select_button = tk.Button(root, text="Select Image", command=select_image)
select_button.pack(pady=10)

root.mainloop()

import cv2 as cv
import numpy as np

def process_image(image_path):
    frame = cv.imread(image_path)

    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Define range of white color in HSV
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([255, 30, 255])

    # Threshold the HSV image to get only white colors
    mask = cv.inRange(hsv, lower_white, upper_white)

    # Bitwise-AND mask and original image
    res = cv.bitwise_and(frame, frame, mask=mask)

    # Display the processed image
    cv.imshow('Original Image', hsv)
    cv.imshow('Mask', mask)
    cv.imshow('Processed Image', res)
    cv.waitKey(0)
    cv.destroyAllWindows()

process_image('/Classification-of-number-of-lanes-transition-areas-and-crossing-from-point-clouds/dataset/2lanes_img/Tile27_.jpg')
