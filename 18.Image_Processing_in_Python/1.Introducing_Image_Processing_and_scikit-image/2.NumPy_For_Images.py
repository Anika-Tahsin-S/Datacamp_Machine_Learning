# Images as NdArrays
import matplotlib.pyplot as plt
madrid_image = plt.imread('/madrid.jpeg')
type(madrid_image)


# Colors with NumPy
# Obtaining red values of the image
red = image[:, :, 0]
# Obtaining green values of the image
green = image[:, :, 1]
# Obtaining blue values of the image
blue = image[:, :, 2]

# Also can display with Grayscale
plt.imshow(red, cmap = "gray")
plt.title("Red")
plt.axis("off")
plt.show()

# Shapes and size
madrid_image.shape
madrid_image.size

# Flip images: vertically
vertically_flipped = np.flipud(madrid_image)
show_image(vertically_flipped, 'Vertically Flipped Image')

# Flip images: horizontally
horizontally_flipped = np.fliplr(madrid_image)
show_image(horizontally_flipped, 'Horizontally Flipped Image')


# Histograms in Matplotlib
# We set bins to 256 because we'll show the number of pixels for every pixel value, that is, from 0 to 255. 
# Meaning we need 256 values to show the histogram. 

red = image[:, :, 0]

plt.hist(red.ravel(), bins = 256)

blue = image[:, :, 2]
plt.hist(blue.ravel(), bins = 256)
plt.title("Blue Histogram")
plt.show()












# --------------------------------------------------------------------------------------------------------- #
##                   Flipping out                  ##
import numpy as np
from skimage import io
flipped_seville = io.imread('Flipping_Out.jpg')

# Part 1
# Flip the image vertically
seville_vertical_flip = np.flipud(flipped_seville)

# Part 2
# Flip the previous image horizontally
seville_horizontal_flip = np.fliplr(seville_vertical_flip)


# Part 3
# Show the resulting image
show_image(seville_horizontal_flip, 'Seville')







##                   Histograms                  ##
import numpy as np
from skimage import io
flipped_seville = io.imread('Histogram_image.jpg')


# Obtain the red channel
red_channel = image[:, :, 0]

# Plot the red histogram with bins in a range of 256
plt.hist(red_channel.ravel(), bins = 256)

# Set title and show
plt.title('Red Histogram')
plt.show()
# With this histogram we see that the image is quite reddish, meaning it has a sensation of warmness. This is because it has a wide and large distribution of bright red pixels, from 0 to around 150.