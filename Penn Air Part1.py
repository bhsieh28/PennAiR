import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from collections import Counter

# Path to image
path = '/Users/bhsieh/Desktop/Penn Air/PennAir 2024 App Static.png'

# Reading image & checking to make sure the path exists
img = cv.imread(path)
assert img is not None, "file could not be read, check with os.path.exists()"

# Calculate the mean of each channel (B, G, R)
average_color_per_channel = np.mean(img, axis=(0, 1))

# Convert the average color to an integer format
average_color = np.round(average_color_per_channel).astype(int) + 70

# Subtract the avg color from each pixel
result = img.astype(int) - average_color

# Clip values to stay within [0, 255] and convert back to uint8
result = np.clip(result, 0, 255).astype(np.uint8)

# cv.imshow('Result Image', result)

lower_bound = np.array([0, 0, 0], dtype=np.uint8)
upper_bound = np.array([36, 36, 36], dtype=np.uint8)
mask = cv.inRange(result, lower_bound, upper_bound)
result[mask > 0] = [0, 0, 0]


# cv.imshow('No noise Image', result)
print(result)
black_pixels = np.where(
    (result[:, :, 0] != 0) | 
    (result[:, :, 1] != 0) | 
    (result[:, :, 2] != 0)
)

# set those pixels to white
result[black_pixels] = [255, 255, 255]

cv.imshow("white shapes image", result)


imgray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# img_hsv = cv.cvtColor(result, cv.COLOR_BGR2HSV)
# contours, _ = cv.findContours(img_hsv, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

fixed_contours = []

for contour in contours:
    if cv.contourArea(contour) > 100:
        fixed_contours.append(contour)

# Iterate over each contour
for contour in fixed_contours:
    # Calculate moments for each contour
    M = cv.moments(contour)
    
    # Calculate the centroid of the contour
    if M["m00"] != 0:  # To avoid division by zero
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0
    
    # Draw a small circle (dot) at the centroid
    cv.circle(img, (cX, cY), 5, (0, 0, 255), -1)  # Red dot with a radius of 5

cv.drawContours(img, fixed_contours, -1, (43,75,238), 3)

cv.imshow("contour image", img)
cv.waitKey(0)