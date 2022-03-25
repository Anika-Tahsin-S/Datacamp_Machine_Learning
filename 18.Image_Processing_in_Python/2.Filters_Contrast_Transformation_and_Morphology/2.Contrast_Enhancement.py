# Histogram equalization
from skimage import exposure

image_eq = exposure.equalize_hist(image)
show_image(image, 'Original')
show_image(image_eq, 'Histogram equalized')


# Adaptive equalization
# CLAHE in scikit-image
from skimage import exposure

image_adapteq = exposure.equalize_adapthist(image, clip_limit = 0.03)
show_image(image, 'Original')
show_image(image_adapteq, 'Adaptive equalized')











# --------------------------------------------------------------------------------------------------------- #
##                  What's the contrast of this image?                  ##
import numpy as np
np.max(clock_image)
# output: 247

np.min(clock_image)
# output: 99

# The contrast is 148. (247 - 99)




##                  Medical images                  ##
import numpy as np
from skimage import io
chest_xray_image = io.imread('Chest_Xray_Image.jpg')

# Part 1
# Import the required module
from skimage import exposure


# Part 2
# Show original x-ray image and its histogram
show_image(chest_xray_image, 'Original x-ray')

plt.title('Histogram of image')
plt.hist(chest_xray_image.ravel(), bins = 256)
plt.show()


# Part 3
# Use histogram equalization to improve the contrast
xray_image_eq =  exposure.equalize_hist(chest_xray_image)


# Part 4
# Show the resulting image
show_image(xray_image_eq, 'Resulting image')






##                  Aerial images                  ##
import numpy as np
from skimage import io
image_aerial = io.imread('Image_Aerial.jpg')

# Import the required module
from skimage import exposure

# Use histogram equalization to improve the contrast
image_eq =  exposure.equalize_hist(image_aerial)

# Show the original and resulting image
show_image(image_aerial, 'Original')
show_image(image_eq, 'Resulting image')














##                  Let's add some impact and contrast                  ##
# Import the necessary modules
from skimage import data, exposure

# Load the image
Coffee_Original = data.coffee()

# Apply the adaptive equalization on the original image
adapthist_eq_image = exposure.equalize_adapthist(Coffee_Original, clip_limit = 0.03)

# Compare the original image to the equalized
show_image(Coffee_Original)
show_image(adapthist_eq_image, '#ImageProcessingDatacamp')