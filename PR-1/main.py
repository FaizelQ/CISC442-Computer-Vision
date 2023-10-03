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
image = cv2.imread(image_file_path, cv2.IMREAD_GRAYSCALE)

def create_gaussian_kernel_3x3(sigma):
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]], dtype=np.float32)
    kernel = kernel / (16.0 * np.pi * sigma**2)  # Normalize by the sum of all elements
    return kernel

# Example usage:
sigma = 1.0  # Adjust the sigma value as needed
gaussian_kernel = create_gaussian_kernel_3x3(sigma)

convolved_image = faiCv.convolve(image, gaussian_kernel)
rgb_image = cv2.cvtColor(convolved_image, cv2.COLOR_GRAY2RGB)
cv2.imwrite("lena-gaussian-blur.png", rgb_image)


