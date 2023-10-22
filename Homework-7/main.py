import cv2
import numpy as np

# Load image in

original_image = cv2.imread("./images/Einstein.jpeg")

# 1) Generate and show four levels of multi-resolution. Use a Gaussian kernel of your choice.
# Using cv2.pyrDown() to downsample our image, this reduces the size and applies a Gaussian kernel
cv2.imwrite(f"./images/Q1-Level-{0}-Einstein.jpeg", original_image)
next_level = original_image
for i in range(1, 4):
    next_level = cv2.pyrDown(next_level)
    cv2.imwrite(f"./images/Q1-Level-{i}-Einstein.jpeg", next_level)