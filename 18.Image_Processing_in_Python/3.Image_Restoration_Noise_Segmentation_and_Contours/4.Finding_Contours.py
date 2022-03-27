## Preparing Image
# Transform the image to 2D grayscale
image = color.rgb2gray(image)

# Binarize image
thresh = threshold_otsu(image)
thresholded_image = image > thresh

# Find contours using scikit-image
from skimage import measure

contours = measure.find_contours(thresholded_image, 0.8)

## Summing it up
# Find contours using scikit-image
from skimage import measure
from skimage.filters import threshold_otsu

image = color.rgb2gray(image)
thresh = threshold_otsu(image)
thresholded_image = image > thresh

contours = measure.find_contours(thresholded_image, 0.8)

# A contours shape
for contour in contours:
    print(contour.shape)








# --------------------------------------------------------------------------------------------------------- #
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def show_image_contour(image, contours):
    plt.figure()
    for n, contour in enumerate(contours):
        plt.plot(contour[:, 1], contour[:, 0], linewidth=3)
    plt.imshow(image, interpolation='nearest', cmap='gray_r')
    plt.title('Contours')
    plt.axis('off')


##                  Contouring shapes                  ##
# Import the modules
from skimage import measure, data

# Obtain the horse image
horse_image = data.horse()

# Find the contours with a constant level value of 0.8
contours = measure.find_contours(horse_image, 0.8)

# Shows the image with contours found
show_image_contour(horse_image, contours)






##                  Find contours of an image that is not binary                  ##
from skimage.io import imread
from skimage import color
from skimage import filters

image_dices = imread('Dices.png')

# Part 1
# Make the image grayscale
image_dice = color.rgb2gray(image_dice)


# Part 2
# Obtain the optimal thresh value
thresh = filters.threshold_otsu(image_dice)


# Part 3
# Apply thresholding
binary = image_dice > thresh


# Part 4
# Find contours at a constant value of 0.8
contours = measure.find_contours(binary, 0.8)

# Show the image
show_image_contour(image_dice, contours)







##                  Count the dots in a dice's image                  ##
# Create list with the shape of each contour
shape_contours = [cnt.shape[0] for cnt in contours]

# Set 50 as the maximum size of the dots shape
max_dots_shape = 50

# Count dots in contours excluding bigger than dots size
dots_contours = [cnt for cnt in contours if np.shape(cnt)[0] < max_dots_shape]

# Shows all contours found 
show_image_contour(binary, contours)

# Print the dice's number
print("Dice's dots number: {}. ".format(len(dots_contours)))

# Output: Dice's dots number: 9. 