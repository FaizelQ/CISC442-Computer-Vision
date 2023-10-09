import yaml
import cv2
import numpy as np
import pr1Functions as faiCv

# Load the YAML configuration file
with open('./config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

# Extract the file location for the image you want to process
# Assuming 'lena.png' is the first entry
lena_file_path = config['file_locations'][0]

# Load the image in grayscale
image = cv2.imread(lena_file_path)

# Problem 1 - convolve
sobel_y_kernel = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]])
sobel_image = faiCv.convolve(image, sobel_y_kernel)
cv2.imwrite("./images/convolve-sobel-lena.png", sobel_image)
# Testing
test_sobel_image = cv2.filter2D(src=image, ddepth=-1, kernel=sobel_y_kernel)
cv2.imwrite("./images/test-convolve-sobel.png", test_sobel_image)

# Problem 2 - reduce
image_downsample = faiCv.reduce(image)
cv2.imwrite("./images/reduce-lena.png", image_downsample)
# Testing
test_blurred_image = cv2.GaussianBlur(image, (3, 3), 0)
test_resized_image = cv2.resize(
    test_blurred_image, (image.shape[1] // 2, image.shape[0] // 2))
cv2.imwrite("./images/test-reduce-lena.png", test_resized_image)

# Problem 3 - expand
image_upsampled = faiCv.expand(image_downsample)
cv2.imwrite("./images/expand-function-lena.png", image_upsampled)
# Testing
test_upsampled = cv2.pyrUp(image_downsample)
cv2.imwrite("./images/test-expand-lena.png", test_upsampled)

# Problem 4 - gaussianPyramid
n = 4  # Number of levels
gaussian_pyramid = faiCv.gaussianPyramid(image, n)
for i in range(n):
    cv2.imwrite(
        f"./images/gaussian-pyramind-level{i + 1}-lena.png", gaussian_pyramid[i])

# Problem 5 - laplacianPyramid
laplacian_pyramid = faiCv.laplacianPyramid(image, n)
for i in range(n):
    cv2.imwrite(
        f"./images/laplacian-pyramind-level{i + 1}-lena.png", laplacian_pyramid[i])

# Problem 6 - reconstruct
reconstructed_image = faiCv.reconstruct(laplacian_pyramid, n)
cv2.imwrite("./images/reconstructed-lena.png", reconstructed_image)

image_difference = faiCv.calculate_mse(image, reconstructed_image)
print(f"Image Difference: {image_difference}")