import numpy as np
import cv2 as cv

def canny_corner(image, filename):
    # Canny Edge Detection and Corners
    edges = cv.Canny(image, 30, 100)

    # Corner
    # Apply Harris Corner Detection
    corners = cv.cornerHarris(image, blockSize= 3, ksize = 5, k = 0.05)

    # Threshold for selecting strong corners
    threshold = 0.001 * corners.max()

    corner_coordinates = np.where(corners > threshold)
    # Draw larger circles around the detected corners
    image_with_corners = cv.cvtColor(image, cv.COLOR_GRAY2BGR)  # Convert to color for blue circles
    for y, x in zip(*corner_coordinates):
        cv.circle(image_with_corners, (x, y), 3, (255, 0, 0), -1)  # Blue circles


    # Save the output images
    cv.imwrite(f"{filename}_edges.jpg", edges)
    cv.imwrite(f"{filename}_corners.jpg", image_with_corners)

# Load the image

original_image = cv.imread("./flower_HW4.jpg", cv.IMREAD_GRAYSCALE)

# 1) Run an in-built edge detector and a corner detector, and produce two output images: (i) image with edges, ii) image with corners.
canny_corner(original_image, "task_1")

# 2) Rotate the original image by 60-degrees and perform (1)
angle = 60
rows, cols = original_image.shape[:2] # Num rows and columns
rotation_matrix = cv.getRotationMatrix2D((cols / 2, rows / 2), angle, 1.0) # This will be the center of our image to rotate upon
rotated_image = cv.warpAffine(original_image, rotation_matrix, (cols, rows))
cv.imwrite("rotated_image.jpg", rotated_image)
canny_corner(rotated_image, "rotated_image")

# 3) Scale the original image by 1.6 in both the x and y-directions and perform (1)
scale_matrix = np.array([[1.6, 0  , 0],
                         [0,   1.6, 0],
                         [0,   0,   1]])
scaled_image = cv.warpPerspective(original_image, scale_matrix,(cols, rows))
cv.imwrite("scaled_image.jpg", scaled_image)
canny_corner(scaled_image, "scaled_image")

# 4) Shear the original image in the x-direction by 1.2 and perform (1)
shear_matrix = np.array([[1, 1.2, 0],
                          [0, 1, 0],
                          [0, 0, 1]])

sheared_img = cv.warpPerspective(original_image, shear_matrix, (int(cols * 1.5), int(rows * 1.5))) # Scaling output image dimensions by 1.5 to see full shear
cv.imwrite("sheared_image_x.jpg", sheared_img)
canny_corner(sheared_img, "sheared_img_x")

# 5) Shear the original image in the y-direction by 1.4 and perform (1)
shear_matrix = np.array([[1, 0, 0],
                          [1.4, 1, 0],
                          [0, 0, 1]])

sheared_img = cv.warpPerspective(original_image, shear_matrix, (int(cols * 1.5), int(rows * 1.5))) # Scaling output image dimensions by 1.5 to see full shear
cv.imwrite("sheared_image_y.jpg", sheared_img)
canny_corner(sheared_img, "sheared_img_y")


