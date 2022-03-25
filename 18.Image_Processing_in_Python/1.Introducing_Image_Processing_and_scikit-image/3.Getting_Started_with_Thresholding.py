# Thresholding
# We compare each pixel to a given threshold value:
#  255 (white) if pixel > thresh value: If the pixel is less than that value, we turn it white. 
#  0 (black) if pixel < thresh value: If it's greater; we turn it black. 


# Apply
# Optimal threshold value
thresh = 127

# Apply thresholding to the imgae
binary = image > thresh

# Show original and threshold
show_image(image, 'Original')
show_image(binary, 'Thresholded')


# Inverted Thresholding
# Optimal threshold value
thresh = 127

# Apply thresholding to the imgae
inverted_binary = image <= thresh

# Show original and threshold
show_image(image, 'Original')
show_image(inverted_binary, 'Inverted Thresholded')


# More thresholding Algorithms
from skimage.filters import try_all_threshold

fig, ax = try_all_threshold(image, verbose = False)
show_plot(fig, ax)



## Optimal thresh value; Global; Uniform background
from skimage.filters import threshold_otsu
# Optimal threshold value
thresh = threshold_otsu(image)

# Apply thresholding to the imgae
binary_global = image > thresh

# Show original and threshold
show_image(image, 'Original')
show_image(binary_global, 'Global Thresholding')



## Optimal thresh value; Local; Uneven background
from skimage.filters import threshold_local
# With this function, we calculate thresholds in small pixel regions surrounding each pixel we are binarizing. 
# So we need to specify a block_size to surround each pixel; 
# also known as local neighborhoods.
block_size = 35
# Optimal threshold value
local_thresh = threshold_local(text_image, block_size, offset = 10)

# Apply thresholding to the imgae
binary_local = text_image > local_thresh

# Show original and threshold
show_image(image, 'Original')
show_image(binary_local, 'Local Thresholding')












# --------------------------------------------------------------------------------------------------------- #
##                   Apply Global Thresholding                  ##
import numpy as np
from skimage import io
chess_pieces_image = io.imread('Apply_Global_Thresholding.jpg')

# Import the otsu threshold function
from skimage.filters import threshold_otsu

# Make the image grayscale using rgb2gray
chess_pieces_image_gray = rgb2gray(chess_pieces_image)

# Obtain the optimal threshold value with otsu
thresh = threshold_otsu(chess_pieces_image_gray)

# Apply thresholding to the image
binary = chess_pieces_image_gray > thresh

# Show the image
show_image(binary, 'Binary image')





##                   When the background isn't that obvious                  ##
page_image = io.imread('Page_Image.jpg')
# Part 1
# Import the otsu threshold function
from skimage.filters import threshold_otsu

# Obtain the optimal otsu global thresh value
global_thresh = threshold_otsu(page_image)

# Obtain the binary image by applying global thresholding
binary_global = page_image > global_thresh

# Show the binary image obtained
show_image(binary_global, 'Global thresholding')


# Part 2
# Import the local threshold function
from skimage.filters import threshold_local

# Set the block size to 35
block_size = 35

# Obtain the optimal local thresholding
local_thresh = threshold_local(page_image, block_size, offset = 10)

# Obtain the binary image by applying local thresholding
binary_local = page_image > local_thresh

# Show the binary image
show_image(binary_local, 'Local thresholding')






##                   Trying other methods                  ##
import matplotlib.pyplot as plt
fruits_image = io.imread('Fruits_Image.jpg')

# Import the try all function
from skimage.filters import try_all_threshold

# Import the rgb to gray convertor function 
from skimage.color import rgb2gray

# Turn the fruits_image to grayscale
grayscale = rgb2gray(fruits_image)

# Use the try all method on the resulting grayscale image
fig, ax = try_all_threshold(grayscale, verbose = False)

# Show the resulting plots
plt.show()





##                   Apply thresholding                  ##
tools_image = io.imread('Tools_Image.jpg')
# Import threshold and gray convertor functions
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray

# Turn the image grayscale
gray_tools_image = rgb2gray(tools_image)

# Obtain the optimal thresh
thresh = threshold_otsu(gray_tools_image)

# Obtain the binary image by applying thresholding
binary_image = gray_tools_image > thresh

# Show the resulting binary image
show_image(binary_image, 'Binarized image')

# By using a global thresholding method, you obtained the precise binarized image. 
# If we would have used local instead nothing would have been segmented. 