import cv2
import pr1FunctionsWorking as faiCv
import numpy as np

# Load image in

original_image = cv2.imread("./images/Einstein.jpeg")

# 1) Generate and show four levels of multi-resolution. Use a Gaussian kernel of your choice.
# Using cv2.pyrDown() to downsample our image, this reduces the size and applies a Gaussian kernel
cv2.imwrite(f"./images/Q1-Level-{0}-Multi-res-Einstein.jpeg", original_image)
next_level = original_image
for i in range(1, 4):
    next_level = cv2.pyrDown(next_level)
    cv2.imwrite(f"./images/Q1-Level-{i}-Multi-Res-Einstein.jpeg", next_level)

# 2) Generate and show four levels of multi-scale. Use the same Gaussian kernel as above.
cv2.imwrite(f"./images/Q2-Level-{0}-Multi-scale-Einstein.jpeg", original_image)
next_level = original_image
for i in range(1, 4):
    next_level = cv2.GaussianBlur(next_level, (5, 5), 0)
    cv2.imwrite(f"./images/Q2-Level-{i}-Multi-scale-Einstein.jpeg", next_level)

# 3) Generate Laplacian planes using a Laplacian kernel of your choice
laplacian = faiCv.laplacianPyramid(original_image, 4)
for i in range(len(laplacian)):
    cv2.imwrite(f"./images/Q3-Laplacian-Level-{i}-Einstein.jpeg", laplacian[i])