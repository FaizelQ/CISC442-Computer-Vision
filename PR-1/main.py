import yaml
import cv2
import numpy as np
import pr1FunctionsWorking as faiCv

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
cv2.imwrite("./images/Q1-convolve-sobel-lena.png", sobel_image)
# Testing
test_sobel_image = cv2.filter2D(src=image, ddepth=-1, kernel=sobel_y_kernel)
cv2.imwrite("./images/test-convolve-sobel.png", test_sobel_image)

# Problem 2 - reduce
image_downsample = faiCv.reduce(image)
cv2.imwrite("./images/Q2-reduce-lena.png", image_downsample)
# Testing
test_blurred_image = cv2.GaussianBlur(image, (3, 3), 0)
test_resized_image = cv2.resize(
    test_blurred_image, (image.shape[1] // 2, image.shape[0] // 2))
cv2.imwrite("./images/test-reduce-lena.png", test_resized_image)

# Problem 3 - expand
image_upsampled = faiCv.expand(image_downsample)
cv2.imwrite("./images/Q3-expand-function-lena.png", image_upsampled)
# Testing
test_upsampled = cv2.pyrUp(image_downsample)
cv2.imwrite("./images/test-expand-lena.png", test_upsampled)

# Problem 4 - gaussianPyramid
n = 4  # Number of levels
gaussian_pyramid = faiCv.gaussianPyramid(image, n)
for i in range(n):
    cv2.imwrite(
        f"./images/Q4-gaussian-pyramind-level{i + 1}-lena.png", gaussian_pyramid[i])

# Problem 5 - laplacianPyramid
laplacian_pyramid = faiCv.laplacianPyramid(image, n)
for i in range(n):
    cv2.imwrite(
        f"./images/Q5-laplacian-pyramind-level{i + 1}-lena.png", laplacian_pyramid[i])

# Problem 6 - reconstruct
reconstructed_image = faiCv.reconstruct(laplacian_pyramid, n)
cv2.imwrite("./images/Q6-reconstructed-lena.png", reconstructed_image)

image_difference = faiCv.calc_mse(image, reconstructed_image)
print(f"Image Difference: {image_difference}")
# Problem 7 - Mosaicing
A1 = cv2.imread("./images/Test_A1.png")
A2 = cv2.imread("./images/Test_A2.png")
mosaic = faiCv.blend_images(A1, A2, 4)
cv2.imwrite("./images/Q7-A.png", mosaic)

B1 = cv2.imread("./images/Test_B1.png")
B2 = cv2.imread("./images/Test_B2.png")
mosaic = faiCv.blend_images(B1, B2, 4)
cv2.imwrite("./images/Q7-B.png", mosaic)

C1 = cv2.imread("./images/Test_C1.png")
C2 = cv2.imread("./images/Test_C2.png")
mosaic = faiCv.blend_images(C1, C2, 4)
cv2.imwrite("./images/Q7-C.png", mosaic)

D1 = cv2.imread("./images/Test_D1.png")
D2 = cv2.imread("./images/Test_D2.png")
mosaic = faiCv.blend_images(D1, D2, 4)
cv2.imwrite("./images/Q7-D.png", mosaic)
