import yaml
import cv2
import numpy as np
import pr1Functions as faiCv

# Load the YAML configuration file
with open('config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

# Extract the file location for the image you want to process
image_file_path = config['file_locations'][0]  # Assuming 'lena.png' is the first entry

# Load the image in grayscale
image = cv2.imread(image_file_path)

# TODO
# Create separate testing file and use asserts


# Problem 1 - convolve
sobel_y_kernel = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])
sobel_image = faiCv.convolve(image, sobel_y_kernel)
cv2.imwrite("./images/convolve-function-lena.png", sobel_image)
# Testing
test_sobel_image = cv2.filter2D(src = image, ddepth = -1, kernel = sobel_y_kernel)
cv2.imwrite("./images/test-filter2D-sobel.png", test_sobel_image)

# Problem 2 - reduce
image_downsample = faiCv.reduce(image)
cv2.imwrite("./images/reduce-function-lena.png", image_downsample)
# Testing
blurred_image = cv2.GaussianBlur(image, (3, 3), 0)
resized_image = cv2.resize(blurred_image, (image.shape[1] // 2, image.shape[0] // 2))
print(np.array_equal(image_downsample, resized_image)) # returns True


# Problem 3 - expand
image_upsampled = faiCv.expand(image_downsample)
cv2.imwrite("./images/expand-function-lena.png", image_upsampled)
# Testing
