import cv2
import numpy as np
import logging as log

#################################################################

# Write a function Convolve (I, H). I is an image of varying size, H is a kernel of varying size.
# The output of the function should be the convolution result that is displayed.

# Function performs convolution on an image with a input kernel


def convolve(I_input, H):
    # Checking if image is appropriate for manipulation
    if I_input is None:
        raise ValueError("Image does not exist")
    elif I_input.ndim < 3:
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

# Function downsamples and image by a factor of 2, after applying Gaussian blurring


def reduce(I_input):
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

# Function upsamples an image by a factor of 2


def expand(I_input):
    # Height, width, RGB channels (channels should be 3 - R, G, B)
    rows, cols, channels = I_input.shape

    I_upsample = cv2.resize(I_input, (rows * 2, cols * 2))

    return I_upsample

#################################################################

# Use the Reduce() function to write the GaussianPyramid(I,n) function, where n is the no. of levels.
# Function constructs a gaussian pyramid with n levels

# Function constructs a gaussian pyramid with n levels


def gaussianPyramid(I_input, n):
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
# Function constructs a laplacian pyramid with n levels


def laplacianPyramid(I, n):

    gaussian = gaussianPyramid(I, n)
    laplacian = [gaussian[n-1]]

    # Construct Laplacian pyramid
    for i in range(n - 1, 0, -1):
        level_up = expand(gaussian[i])

        # Difference of Gaussian
        height, width, channels = gaussian[i-1].shape
        level_up = cv2.resize(level_up, (width, height))
        difference = cv2.subtract(gaussian[i-1], level_up)
        laplacian.append(difference)

    # Highest level of pyramid
    laplacian.append(gaussian[n - 1])
    return laplacian

#################################################################

# Write the Reconstruct(LI,n) function which collapses the Laplacian pyramid LI of n levels
# to generate the original image. Report the error in reconstruction using image difference.

# Function reconstructs and image to a single matrix using Laplacian pyramids.


def reconstruct(LI, n):

    # We want to tackle the pyramid in the opposite direction for reconstruciton
    I_output = LI[0]

    for i in range(1, n):
        # Upsammple image with exapnd for addition
        upsampled = expand(I_output)
        # Add the image for expansion
        I_output = cv2.add(upsampled, LI[i])

    return I_output

#################################################################

# Function calculates the Mean Squared Error of the image difference between the original image and reconstructed image.


def calc_mse(image_1, image_2):
    # Convert the images to grayscale.
    image_1_gray = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    image_2_gray = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

    # Compute the pixel difference between the two images.
    difference = image_1_gray - image_2_gray

    # Square the pixel differences.
    squared_difference = difference ** 2

    # Compute the average of the squared pixel differences.
    mse = np.mean(squared_difference)

    return mse

# Function registers mouse click for coordinates


def mouse_coordinates_boundary(image1, image2):

    resulting_coordinates = []

    # Copy image
    images = [image1.copy(), image2.copy()]

    for image in images:
        cv2.namedWindow('Image Window')

        # Mouse click handler
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                resulting_coordinates.append((x, y))
                cv2.circle(image, (x, y), 3, (255, 0, 0), -1)
                cv2.imshow('Image Window', image)

        # Set handler
        cv2.setMouseCallback('Image Window', mouse_callback)

        cv2.imshow('Image Window', image)
        cv2.waitKey(0)
        cv2.destroyWindow('Image Window')

    return resulting_coordinates


def resize_and_pad(left, right, boundaries):

    left_height, left_width, l_channels = left.shape
    right_height, right_width, r_channels = right.shape

    # Scalar value
    s = left_height / right_height

    right = cv2.resize(right, (right_width, int(s*right_height)))

    overlapping_pixels_1 = left_width - boundaries[0][0]
    overlapping_pixels_2 = boundaries[1][0]

    output_width = left_width + right_width - overlapping_pixels_1 - overlapping_pixels_2

    padding_right = max(0, output_width - left_width)
    padding_left = max(0, output_width - right_width)

    # Pad images inorder for alignment and output image
    resized_left = cv2.copyMakeBorder(
        src=left, top=0, bottom=0, left=0, right=padding_right, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
    resized_right = cv2.copyMakeBorder(
        src=right, top=0, bottom=0, left=padding_left, right=0, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))

    cv2.destroyAllWindows()

    return resized_left, resized_right

# Function blends two images, left and right by creating n level bitmask layers


def blend_images(left, right, n):

    # Assume boundaries
    boundaries = mouse_coordinates_boundary(left, right)
    resized_left, resized_right = resize_and_pad(left, right, boundaries)

    laplacianPyramid_left = laplacianPyramid(resized_left, n)
    laplacianPyramid_right = laplacianPyramid(resized_right, n)

    # Zip left and right images (add images)
    Layers = []
    for l, r in zip(laplacianPyramid_left, laplacianPyramid_right):
        rows, cols, dpt = l.shape
        current_layer = np.hstack((l[:, 0:cols//2], r[:, cols//2:]))
        Layers.append(current_layer)

    # Perform reconstruction
    reconstructed = Layers[0]
    for i in range(1, n):
        height, width, channels = Layers[i].shape
        reconstructed = cv2.resize(reconstructed, (width, height))
        reconstructed = cv2.add(reconstructed, Layers[i])

    return reconstructed
