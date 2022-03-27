# Apply noise in scikit-image
from skimage.util import random_noise

noisy_image = random_noise(dog_image)
show_image(dog_image)
show_image(noisy_image, 'Noisy Image')


## Denoising
# Using total variation filter denoising
from skimage.restoration import denoise_tv_chambolle

denoised_image = denoise_tv_chambolle(noisy_image, weigth = 0.1, multichannel = True)
show_image(noisy_image, 'Noisy Image')
show_image(denoised_image, 'Denoised Image')



# Bilateral filter denoising
from skimage.restoration import denoise_bilateral

denoised_image = denoise_bilateral(noisy_image, multichannel = True)
show_image(noisy_image, 'Noisy Image')
show_image(denoised_image, 'Denoised Image')








# --------------------------------------------------------------------------------------------------------- #
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import io
fruit_image = io.imread('Fruit_Image.jpg')
landscape_image = io.imread('Noise_Noisy_Nature.jpg')


##                  Let's make some noise!                  ##
# Import the module and function
from skimage.util import random_noise

# Add noise to the image
noisy_image = random_noise(fruit_image)

# Show original and resulting image
show_image(fruit_image, 'Original')
show_image(noisy_image, 'Noisy image')





##                  Reducing noise                  ##
# Import the module and function
from skimage.restoration import denoise_tv_chambolle

# Apply total variation filter denoising
denoised_image = denoise_tv_chambolle(noisy_image, multichannel = True)

# Show the noisy and denoised images
show_image(noisy_image, 'Noisy')
show_image(denoised_image, 'Denoised image')





##                  Reducing noise while preserving edges                  ##
# Import bilateral denoising function
from skimage.restoration import denoise_bilateral

# Apply bilateral filter denoising
denoised_image = denoise_bilateral(landscape_image,  multichannel = True)

# Show original and resulting images
show_image(landscape_image, 'Noisy image')
show_image(denoised_image, 'Denoised image')