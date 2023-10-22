import cv2
import hw7functions as faiCv

# Load image in

original_image = cv2.imread("./images/Einstein.jpeg")

# 1) Generate and show four levels of multi-resolution. Use a Gaussian kernel of your choice.
# Using cv2.pyrDown() to downsample our image, this reduces the size and applies a Gaussian kernel
mrGaussianPyramid = faiCv.multi_res(original_image, 4)

# 2) Generate and show four levels of multi-scale. Use the same Gaussian kernel as above.
cv2.imwrite(f"./images/Q2-Level-{0}-Multi-scale-Einstein.jpeg", original_image)
next_level = original_image
for i in range(1, 4):
    next_level = cv2.GaussianBlur(next_level, (5, 5), 0)
    cv2.imwrite(f"./images/Q2-Level-{i}-Multi-scale-Einstein.jpeg", next_level)

# 3) Generate Laplacian planes using a Laplacian kernel of your choice
laplacian_pyramid = faiCv.laplacianPyramid(original_image, 4)
# 4) Generate an approximation to Laplacian using the difference of Gaussian planes from (1). Note: you need to do 'Expand' on images before taking the difference.
q4_laplacian_pyramid = faiCv.q4_laplacianPyramid(original_image, 4)
# 5) Generate an approximation to Laplacian using the difference of Gaussian planes from (2)
q5_laplacian_pyramid = faiCv.q5_laplacianPyramid(original_image, 4)
