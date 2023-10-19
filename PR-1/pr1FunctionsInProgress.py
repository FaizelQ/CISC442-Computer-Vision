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
def laplacianPyramid(I_input, n):
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
    laplacian_pyramid.append(gaussian_pyramid[-1])

    return laplacian_pyramid


#################################################################

# Write the Reconstruct(LI,n) function which collapses the Laplacian pyramid LI of n levels
# to generate the original image. Report the error in reconstruction using image difference.

# Function reconstructs and image to a single matrix using Laplacian pyramids.
def reconstruct(LI, n):

    # Initalize output image as highest level of pyramid
    I_output = LI[n - 1]

    # Iterate backwards from Top to Bottom collapsing
    for i in range(n - 2, -1, -1):
        # Upsample current level for matrix addition and so dimensions match
        I_output = expand(I_output)
        I_output = cv2.add(I_output, LI[i])

    return I_output

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

#################################################################

# Finally, you will be mosaicking images using Laplacian plane based reconstruction (Note that your program
# should handle color images). Let the user pick the blend boundaries of all images by mouse. Blend
# the left image with the right image. Note the left and right images share a joint region. Submit four results of
# mosaicking. Each mosaic can be comprised of 2 individual images from different cameras/viewpoints.

# Helper function to resize and pad images for blending using copyMakeBorder


def resize_and_pad(left_image, right_image, x, overlap_region):
    resized_left = np.empty((0, 0, 3), dtype=np.uint8)
    resized_right = np.empty((0, 0, 3), dtype=np.uint8)
    # Left and right image dimensions/shape
    l_height, l_width, l_rgb = left_image.shape
    r_height, r_width, r_rgb = right_image.shape

    # Resize left image
    # If right image is bigger than left image, resize left image by adding padding to bottom
    if r_height - l_height > 0:
        resized_left = cv2.copyMakeBorder(
            src=left_image, top=0, bottom=r_height - l_height, left=0, right=r_width - overlap_region, borderType=0)
    # If left image is bigger, only create add padding for overlap region to resize
    else:
        resized_left = cv2.copyMakeBorder(
            src=left_image, top=0, bottom=0, left=0, right=r_width - x, borderType=0)

    # Resize right image
    # If left image is bigger than right image, resize left image by adding padding to bottom
    # Create padding for left size of right image with padding
    if l_height - r_height > 0:
        resized_right = cv2.copyMakeBorder(
            src=right_image, top=0, bottom=l_height - r_height, left=x, right=0, borderType=0)
    # If right image is bigger, only create add padding for overlap region to resize
    else:
        resized_right = cv2.copyMakeBorder(
            src=right_image, top=0, bottom=0, left=x, right=0, borderType=0)

    return resized_left, resized_right
# Function blends two images together with n_levels for pyramids and uses mouse click events for blend boundary


def blend_images(left_image, right_image, n_levels):

    # Helper and handler function for registering mouse click coordinates
    def mouse_handler_onClick(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"PRINTING x: {x}")
            print(f"PRINTING x, y {(x,y)}")
            #x_resize = x
            #cv2.destroyAllWindows()
            # Image width - coordinate of mouse click
            overlap_region = left_image.shape[1] - x
            # Resize images
            # resized_left, resized_right = resize_and_pad(
            #     left_image, right_image, x, overlap_region)
            resized_left = cv2.copyMakeBorder(left_image, 0, right_image.shape[0] - left_image.shape[0] if right_image.shape[0] - left_image.shape[0] > 0 else 0, 0, right_image.shape[1] - overlap_region, 0)
            resized_right = cv2.copyMakeBorder(right_image, 0, left_image.shape[0] - right_image.shape[0] if left_image.shape[0] - right_image.shape[0] > 0 else 0, x, 0, 0)
            # Create bitmask - max(left vs right height) x right width x RGB
            bitmask = np.empty(
                (max(left_image.shape[0], right_image.shape[0]), right_image.shape[1] + x, left_image.shape[2]))
            bitmask[:, x: x + overlap_region] = [0.5] * left_image.shape[2]
            bitmask[:, x + overlap_region:] = [1] * left_image.shape[2]

            left_image_laplacian = laplacianPyramid(resized_left, n_levels)
            right_image_laplacian = laplacianPyramid(resized_right, n_levels)
            bitmask_gp = gaussianPyramid(bitmask, n)

            laplacian_result = []
            for i in range(n_levels):
                layer = cv.multiply(np.float64(left_image_laplacian[i]), (
                    1 - bitmask_gp[i])) + cv.multiply(np.float64(right_image_laplacian[i]), bitmask_gp[i])
                layer[layer > 255] = 255
                layer[layer < 0] = 0
                layer = np.uint8(layer)

                laplacian_result.append(layer)
            image_output = reconstruct(laplacian_result, n_levels)
            cv2.imwrite("./images/Q7-Blend.png", image_output)
            return
        else:
            return

    print("Mark boundary on left image (one click)")
    cv2.namedWindow('Image Window')
    cv2.setMouseCallback('Image Window', mouse_handler_onClick)
    cv2.imshow('Image Window', left_image)
    cv2.waitKey(0)
    print(x)
