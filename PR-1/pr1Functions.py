import cv2
import numpy as np
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
    padded_image = cv2.copyMakeBorder(
        I_input, padding_length, padding_length, padding_length, padding_length, cv2.BORDER_CONSTANT)

    # Create output image of same size (using np.empty instead of np.zeros to save computational time/power)
    I_output = np.empty(I_input.shape)

    # Loop through rows, cols, and each color pixel channel
    for i in range(rows):
        for j in range(cols):
            for c in range(channels):
                # np.reshape(M, -1) will flatten our matrix to a 1D array
                # padded_image[i : i + kernel_cols, j : j + kernel_rows, k], slicing our array. [0 : 3, 0 : 3] will go from rows 0 (inclusive) to 3 (exluded)
                I_output[i][j][c] = np.reshape(
                    H, -1) @ np.reshape(padded_image[i: i + kernel_cols, j: j + kernel_rows, c], -1)

    return I_output

#################################################################

# Write a function Reduce(I) that takes image I as input and outputs a copy of the image resampled
# by half the width and height of the input. Remember to Gaussian filter the image before reducing it;
# use separable 1D Gaussian kernels.


def reduce(I_input):

    # Checking if image is appropriate for manipulation
    if I_input is None:
        raise ValueError("Image does not exist")
    elif I_input.ndim < 2:
        raise ValueError("Image must be 2D matrix")
    else:
        log.info("Image is exists and is valid")

    # Height, width, RGB channels (channels should be 3 - R, G, B)
    rows, cols, channels = I_input.shape

    # Apply Gaussian blurring to image before downsampling
    I_blurred = cv2.GaussianBlur(I_input, (3, 3), 0)

    # Resize the filtered image to half its width and height
    I_downsample = cv2.resize(I_blurred, (rows // 2, cols // 2))

    return I_downsample

#################################################################

# Write a function Expand(I) that takes image I as input and outputs a copy of the image expanded,
# twice the width and height of the input.


def expand(I_input):
    # Checking if image is appropriate for manipulation
    if I_input is None:
        raise ValueError("Image does not exist")
    elif I_input.ndim < 2:
        raise ValueError("Image must be 2D matrix")
    else:
        log.info("Image is exists and is valid")

    # Height, width, RGB channels (channels should be 3 - R, G, B)
    rows, cols, channels = I_input.shape

    I_upsample = cv2.resize(I_input, (rows * 2, cols * 2))

    return I_upsample

#################################################################

# Use the Reduce() function to write the GaussianPyramid(I,n) function, where n is the no. of levels.


def gaussianPyramid(I_input, n):
    # Checking if image is appropriate for manipulation
    if I_input is None:
        raise ValueError("Image does not exist")
    elif I_input.ndim < 2:
        raise ValueError("Image must be 2D matrix")
    else:
        log.info("Image is exists and is valid")

    # Create list to hold each level of pyramid
    gaussian_pyramind = []
    gaussian_pyramind.append(I_input)

    # Iterate through each level of pyramid and reduce
    next_level = I_input
    for i in range(n):
        next_level = reduce(next_level)
        gaussian_pyramind.append(next_level)

    return gaussian_pyramind

#################################################################

# Use the above functions to write LaplacianPyramids(I,n) that produces n level Laplacian pyramid of I.


def laplacianPyramid(I_input, n):
    # Checking if image is appropriate for manipulation
    if I_input is None:
        raise ValueError("Image does not exist")
    elif I_input.ndim < 2:
        raise ValueError("Image must be 2D matrix")
    else:
        log.info("Image is exists and is valid")

    # Construct Gaussian pyramid
    gaussian_pyramid = gaussianPyramid(I_input, n)

    # Construct Laplacian pyramid
    laplacian_pyramid = []

    # Bottom to top construction
    for i in range(n - 1):
        level_up = expand(gaussian_pyramid[i + 1])
        difference = cv2.subtract(gaussian_pyramid[i], level_up)
        laplacian_pyramid.append(difference)
    # Ln = Gn (Top level)
    laplacian_pyramid.append(gaussian_pyramid[n - 1])

    return laplacian_pyramid


#################################################################

# Write the Reconstruct(LI,n) function which collapses the Laplacian pyramid LI of n levels
# to generate the original image. Report the error in reconstruction using image difference.


def reconstruct(LI, n):

    # Initalize output image as highest level of pyramid
    I_output = LI[n - 1]

    # Iterate backwards from Top to Bottom collapsing
    for i in range(n - 2, -1, -1):
        # Upsample current level for matrix addition and so dimensions match
        I_output = expand(I_output)
        I_output += LI[i]
    
    return I_output

#################################################################
