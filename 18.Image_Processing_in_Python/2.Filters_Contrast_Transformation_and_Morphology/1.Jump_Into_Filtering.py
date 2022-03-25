# Egde Detection
# Sobel
from skimage.filters import sobel

edge_sobel = sobel(image_coins)
plot_comparison(image_coins, edge_sobel, "Edge with Sobel")

# Comparing plots
def plot_comparison(original, filtered, title_filtered):
    fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize = (8, 6), sharex = True, sharey = False)
    ax1.imshow(original, cmap = plt.cm.gray)
    ax1.set_title('original')
    ax1.axis('off')
    ax2.imshow(filtered, cmap = plt.cm.gray)
    ax2.set_title(title_filtered)
    ax2.axis('off')


# Gaussian smoothing
from skimage.filters import gaussian
gaussian_image = gaussian(amsterdam_pic, multichannel = True)
plot_comparisson(amsterdam_pic, gaussian_image, "Blurred with Gaussian Filter")











# --------------------------------------------------------------------------------------------------------- #
##                   Edge detection                  ##
import numpy as np
from skimage import io
soaps_image = io.imread('Soaps_Image.jpg')


# Import the color module
from skimage import color

# Import the filters module and sobel function
from skimage.filters import sobel

# Make the image grayscale
soaps_image_gray = color.rgb2gray(soaps_image)

# Apply edge detection filter
edge_sobel = sobel(soaps_image_gray)

# Show original and resulting image to compare
show_image(soaps_image, "Original")
show_image(edge_sobel, "Edges with Sobel")






##                   Blurring to reduce noise                  ##
building_image = io.imread('Building_Image.jpg')
# Import Gaussian filter 
from skimage.filters import gaussian

# Apply filter
gaussian_image = gaussian(building_image, multichannel = True)

# Show original and resulting image to compare
show_image(building_image, "Original")
show_image(gaussian_image, "Reduced sharpness Gaussian")

plot_comparison(building_image, gaussian_image, "Reduced sharpness Gaussian")