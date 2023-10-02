import cv2 as cv
import numpy as np
import os

#################################################################

# Write a function Convolve (I, H). I is an image of varying size, H is a kernel of varying size.
# The output of the function should be the convolution result that is displayed.

def convolve(I, H):
    # Check if H (Kernel) is appropriate size
    if H.ndim != 2:
        raise ValueError("Kernel must be a 2-D Matrix")

    # Get Dimensions of both input image and kernel
    rows, cols = I.shape
    kernel_rows, kernel_cols = H.shape

    # Calculate padding for input and output to have same size image
    padding_top = kernel_rows // 2 # 3x3 will return 1
    padding_bottom = kernel_rows - padding_top - 1
    padding_left = kernel_cols // 2
    padding_right = kernel_cols - padding_left - 1

    # Create padded image based off input image
    I_padded = np.pad(I, ((padding_top, padding_bottom), (padding_left, padding_right)), mode='constant')

    # Create an empty output grayscale image
    # Matrix of all zeroes
    # 8 bit image (0-255) values
    Iout = np.zeros((rows, cols), dtype=np.uint8)

    # Loop through each pixel in the output image
    for i in range(rows):
        for j in range(cols):
            # Calculate the coordinates of the central pixel of the kernel
            i_kernel_center = i + kernel_rows // 2
            j_kernel_center = j + kernel_cols // 2

            # Extract the kernel centered at (i_kernel_center, j_kernel_center) from the padded image
            i_kernel_start = i_kernel_center - kernel_rows // 2
            i_kernel_end = i_kernel_center + kernel_rows // 2 + 1
            j_kernel_start = j_kernel_center - kernel_cols // 2
            j_kernel_end = j_kernel_center + kernel_cols // 2 + 1
            working_kernel = I_padded[i_kernel_start:i_kernel_end, j_kernel_start:j_kernel_end]
            # For example, [0:3, 0:3] slicing would be the working kernel for (0, 0)
            # Select rows 0 (inclusive) up to 3, and select cols 0 (inclusive) up to 3

            # Element-wise multiplication and sum to compute the convolution result of the pixel
            Iout[i, j] = np.sum(working_kernel * H)

    # Display the convolved grayscale image
    cv.imshow('Convolved Grayscale Image', Iout)
    cv.waitKey(0)
    cv.destroyAllWindows()
#################################################################

# Write a function Reduce(I) that takes image I as input and outputs a copy of the image resampled
# by half the width and height of the input. Remember to Gaussian filter the image before reducing it; 
# use separable 1D Gaussian kernels.

def reduce(I):


#################################################################

# Write a function Expand(I) that takes image I as input and outputs a copy of the image expanded, 
# twice the width and height of the input.

def expand(I):


#################################################################

# Use the Reduce() function to write the GaussianPyramid(I,n) function, where n is the no. of levels.

def gaussianPyramid(I, n):


#################################################################

# Use the above functions to write LaplacianPyramids(I,n) that produces n level Laplacian pyramid of I.

def laplacianPyramid(I, n):


#################################################################

# Write the Reconstruct(LI,n) function which collapses the Laplacian pyramid LI of n levels 
# to generate the original image. Report the error in reconstruction using image difference.

def reconstruct(LI, n):


#################################################################

