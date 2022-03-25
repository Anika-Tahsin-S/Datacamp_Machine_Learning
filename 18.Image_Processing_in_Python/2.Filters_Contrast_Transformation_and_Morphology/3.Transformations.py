# Rotating Clockwise
from skimage.transform import rotate

image_rotate = rotate(image, -90)

show_image(image, 'Original')
show_image(image_rotated, 'Rotated 90 degrees clockwise')


# Rotating Anticlockwise
from skimage.transform import rotate

image_rotate = rotate(image, 90)

show_image(image, 'Original')
show_image(image_rotated, 'Rotated 90 degrees clockwise')


# Rescaling; Downgrading
from skimage.transform import rescale

image_rescaled = rescale(image, 1/4, anti_aliasing = True, multichannel = True)

show_image(image, 'Original')
show_image(image_rescaled, 'Rescaled Image')



# Resizing
from skimage.transform import resize

height = 400
width = 500

image_resized = resize(image, (height, width), anti_aliasing = True)

show_image(image, 'Original')
show_image(image_resized, 'Resized Image')




# Resizing Proportionally
from skimage.transform import resize

height = image.shape[0] / 4
width = image.shape[1] / 4

image_resized = resize(image, (height, width), anti_aliasing = True)

show_image(image, 'Original')
show_image(image_resized, 'Resized Image')










# --------------------------------------------------------------------------------------------------------- #
##                  Aliasing, rotating and rescaling                  ##
import numpy as np
from skimage import io
image_cat = io.imread('Image_Cat.jpg')

# Part 1
# Import the module and the rotate and rescale functions
from skimage.transform import rotate, rescale


# Part 2
# Rotate the image 90 degrees clockwise 
rotated_cat_image = rotate(image_cat, -90)

# Part 3
# Rescale with anti aliasing
rescaled_with_aa = rescale(rotated_cat_image, 1/4, anti_aliasing = True, multichannel = True)


# Part 4
# Rescale without anti aliasing
rescaled_without_aa = rescale(rotated_cat_image, 1/4, anti_aliasing = False, multichannel = True)

# Show the resulting images
show_image(rescaled_with_aa, "Transformed with anti aliasing")
show_image(rescaled_without_aa, "Transformed without anti aliasing")








##                  Enlarging images                  ##
# Import the module and function to enlarge images
from skimage.transform import rescale

# Import the data module
from skimage import data

# Load the image from data
rocket_image = data.rocket()

# Enlarge the image so it is 3 times bigger
enlarged_rocket_image = rescale(rocket_image, 3, anti_aliasing = True, multichannel = True)

# Show original and resulting image
show_image(rocket_image)
show_image(enlarged_rocket_image, "3 times enlarged image")







##                  Proportionally resizing                  ##
dogs_banner = io.imread('Dogs_Banner.jpg')

# Import the module and function
from skimage.transform import resize

# Set proportional height so its half its size
height = int(dogs_banner.shape[0] / 2)
width = int(dogs_banner.shape[1] / 2)

# Resize using the calculated proportional height and width
image_resized = resize(dogs_banner, (height, width), anti_aliasing = True)

# Show the original and resized image
show_image(dogs_banner, 'Original')
show_image(image_resized, 'Resized image')