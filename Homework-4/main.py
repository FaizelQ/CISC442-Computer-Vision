import numpy as np
import cv2 as cv

def canny_corner(image, filename):
    # Canny Edge Detection and Corners
    edges = cv.Canny(image, 100, 150)

    # Corner
    # Apply Harris Corner Detection
    corners = cv.cornerHarris(image, blockSize= 2, ksize = 3, k = 0.04)

    # Threshold for selecting strong corners
    threshold = 0.001 * corners.max()

    corner_coordinates = np.where(corners > threshold)
    # Draw larger circles around the detected corners
    image_with_corners = cv.cvtColor(image, cv.COLOR_GRAY2BGR)  # Convert to color for blue circles
    for y, x in zip(*corner_coordinates):
        cv.circle(image_with_corners, (x, y), 1, (255, 0, 0), -1)  # Blue circles


    # Save the output images
    cv.imwrite(f"{filename}_edges.jpg", edges)
    cv.imwrite(f"{filename}_corners.jpg", image_with_corners)

# Load the image

original_image = cv.imread("./flower_HW4.jpg", cv.IMREAD_GRAYSCALE)

# 1) Run an in-built edge detector and a corner detector, and produce two output images: (i) image with edges, ii) image with corners.
canny_corner(original_image, "task_1")

# 2) Rotate the original image by 60-degrees and perform (1)
angle = 60
height, width = original_image.shape[:2] # Num rows and columns
rotation_matrix = cv.getRotationMatrix2D((width / 2, height / 2), angle, 1.0) # This will be the center of our image to rotate upon
rotated_image = cv.warpAffine(original_image, rotation_matrix, (width, height))
cv.imwrite("rotated_image.jpg", rotated_image)
canny_corner(rotated_image, "rotated_image")

# 3) Scale the original image by 1.6 in both the x and y-directions and perform (1)




