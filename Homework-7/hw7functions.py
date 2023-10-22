import cv2

'''
Creates multi-res gaussian pyramid with n levels using cv2.pyrDown() and returns list of images
'''


def multi_res(I_input, n_levels):
    mrGaussianPyramid = [I_input]

    cv2.imwrite(f"./images/Q1-Level-{0}-Multi-res-Einstein.jpeg", I_input)
    next_level = I_input

    for i in range(1, 4):
        next_level = cv2.pyrDown(next_level)
        mrGaussianPyramid.append(next_level)
        cv2.imwrite(f"./images/Q1-Level-{i}-Multi-Res-Einstein.jpeg", next_level)

    return mrGaussianPyramid


'''
Creates multi-scale gaussian pyramid with n levels using cv2.GaussianBlur()
'''


def multi_scale(I_input, n_levels):
    msGaussianPyramid = [I_input]

    cv2.imwrite(f"./images/Q2-Level-{0}-Multi-scale-Einstein.jpeg", I_input)
    next_level = I_input
    for i in range(1, 4):
        next_level = cv2.GaussianBlur(next_level, (5, 5), 0)
        msGaussianPyramid.append(next_level)
        cv2.imwrite(f"./images/Q2-Level-{i}-Multi-scale-Einstein.jpeg", next_level)

    return msGaussianPyramid


'''
Creates lapclian pyramid from our multi_scale images
'''


def laplacianPyramid(I_input, n_levels):
    gaussian_pyramid = multi_res(I_input, n_levels)
    laplacian_pyramid = []

    for i in range(0, n_levels - 1):
        height, width, channels = gaussian_pyramid[i].shape
        resized = cv2.resize((gaussian_pyramid[i + 1]), (width, height))
        laplace_level = cv2.subtract(gaussian_pyramid[i], resized)
        laplacian_pyramid.append(laplace_level)
        cv2.imwrite(f"./images/Q3-Laplacian_Level-{i}-Einstein.jpeg", laplace_level)

    laplacian_pyramid.append(gaussian_pyramid[n_levels - 1])
    cv2.imwrite(f"./images/Q3-Laplacian_Level-{n_levels - 1}-Einstein.jpeg", laplacian_pyramid[n_levels - 1])
    return laplacian_pyramid


'''
Function is same as function above, except the cv2.imwrite path names are difference
'''


def q4_laplacianPyramid(I_input, n_levels):
    gaussian_pyramid = multi_res(I_input, n_levels)
    laplacian_pyramid = []

    for i in range(0, n_levels - 1):
        height, width, channels = gaussian_pyramid[i].shape
        resized = cv2.resize((gaussian_pyramid[i + 1]), (width, height))
        laplace_level = cv2.subtract(gaussian_pyramid[i], resized)
        laplacian_pyramid.append(laplace_level)
        cv2.imwrite(f"./images/Q4-Laplacian_Level-{i}-Einstein.jpeg", laplace_level)

    laplacian_pyramid.append(gaussian_pyramid[n_levels - 1])
    cv2.imwrite(f"./images/Q4-Laplacian_Level-{n_levels - 1}-Einstein.jpeg", laplacian_pyramid[n_levels - 1])
    return laplacian_pyramid


def q5_laplacianPyramid(I_input, n_levels):
    gaussian_pyramid = multi_scale(I_input, n_levels)
    laplacian_pyramid = []

    for i in range(0, n_levels - 1):
        laplace_level = cv2.subtract(gaussian_pyramid[i], gaussian_pyramid[i + 1])
        laplacian_pyramid.append(laplace_level)
        cv2.imwrite(f"./images/Q5-Laplacian_Level-{i}-Einstein.jpeg", laplace_level)

    laplacian_pyramid.append(gaussian_pyramid[n_levels - 1])
    cv2.imwrite(f"./images/Q5-Laplacian_Level-{n_levels - 1}-Einstein.jpeg", laplacian_pyramid[n_levels - 1])
    return laplacian_pyramid
