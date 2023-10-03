import cv2
import numpy as np
import os
import logging as log

#################################################################

# Write a function Convolve (I, H). I is an image of varying size, H is a kernel of varying size.
# The output of the function should be the convolution result that is displayed.


def convolve(I_input, H):

    # Checking if image is appropriate for manipulation
    if I_input is None:
        raise ValueError("Image does not exist")
    elif I_input.ndim < 2:
        raise ValueError("Image must be 2D matrix")
    else:
        log.info("Image is exists and is valid")

    # Height, width, RGB channels (channels should be 3 - R, G, B)
    rows, cols, channels = I_input.shape

    # Height, width of kernel H, also assuming that Kernel is m x m, ex. 3x3
    kernel_rows, kernel_cols = H.shape

    # Padding will be needed for convolution, ex. 3x3, padding_length = 1
    padding_length = kernel_rows // 2
    padded_image = cv2.copyMakeBorder(I_input, padding_length, padding_length, padding_length, padding_length, cv2.BORDER_CONSTANT)

    # Create output image of same size (using np.empty instead of np.zeros to save computational time/power)
    I_output = np.empty(I_input.shape)

    # Loop through rows, cols, and each color pixel channel
    


#################################################################

# Write a function Reduce(I) that takes image I as input and outputs a copy of the image resampled
# by half the width and height of the input. Remember to Gaussian filter the image before reducing it;
# use separable 1D Gaussian kernels.

def reduce(I):
    pass

#################################################################

# Write a function Expand(I) that takes image I as input and outputs a copy of the image expanded,
# twice the width and height of the input.


def expand(I):
    pass


#################################################################

# Use the Reduce() function to write the GaussianPyramid(I,n) function, where n is the no. of levels.

def gaussianPyramid(I, n):
    pass

#################################################################

# Use the above functions to write LaplacianPyramids(I,n) that produces n level Laplacian pyramid of I.


def laplacianPyramid(I, n):
    pass

#################################################################

# Write the Reconstruct(LI,n) function which collapses the Laplacian pyramid LI of n levels
# to generate the original image. Report the error in reconstruction using image difference.


def reconstruct(LI, n):
    pass

#################################################################
