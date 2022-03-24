# There are some testing-purpose images provided by scikit-image, in a module called data. 
# If we want to load a colored image of a rocket, we can do so by: Importing data from skimage.
# Call a method .rocket()

from skimage import data
rocket_image = data.rocket()

# RGB VS Grayscale
from skimage import color
grayscale = color.rgb2gray(original)
rgb = color.gray2rgb(grayscale)

# Visualizing images
import matplotlib.pyplot as plt
def show_image(image, title = 'Image', cmap_type = 'gray'):
    plt.imshow(image, cmap = camp_type)
    plt.title(title)
    plt.axis('off')
    plt.show()

from skimage import color
grayscale = color.rgb2gray(original)
show_image(grayscale, "Grayscale")









# --------------------------------------------------------------------------------------------------------- #
##                   Is this gray or full of color?                  ##
# Whats the main difference between the images shown below?
import numpy as np
from skimage import data
coffee_image = data.coffee()
coins_image = data.coins()

print(coffee_image.shape)
# (400, 600, 3)
print(coins_image.shape)
# (303, 384)
or, 
np.shape(coffee_image)
np.shape(coins_image)


# Answer: coins_image has a shape of (303, 384), grayscale. And coffee_image (400, 600, 3), RGB-3.
# The coffee image is RGB-3 colored, that's why it has a 3 at the end, when displaying the shape (H, W, D) of it. While the coins image is grayscale and has a single color channel.






##                   RGB to grayscale                  ##
# Preloaded
show_image(image, title = 'Image')

# Import the modules from skimage
from skimage import data, color

# Load the rocket image
rocket = data.rocket()

# Convert the image to grayscale
gray_scaled_rocket = color.rgb2gray(rocket)

import matplotlib.pyplot as plt
def show_image(image, title = 'Image', cmap_type = 'gray'):
    plt.figure(figure_size = (640, 480))
    plt.imshow(image, cmap = camp_type)
    plt.title(title)
    plt.axis('off')
    plt.show()
# plt.figure()
# <Figure size 640x480 with 0 Axes>


# Show the original image
show_image(rocket, 'Original RGB image')

# Show the grayscale image
show_image(gray_scaled_rocket, 'Grayscale image')