import numpy as np
import matplotlib.pyplot as plt

# Harris corner Detector
from skimage.feature import corner_harris
image = rgb2gray(image)
measure_image = corner_harris(image)

show_image(measure_image)

# Finding coordinates
coords = corner_peaks(corner_harris(image), min_distance = 5)

print("A total of", len(coords), "corners were detected.")

def show_image_with_corners(image, coords, title="Corners detected"):
    plt.imshow(image, interpolation='nearest', cmap = 'gray')
    plt.title(title)
    plt.plot(coords[:, 1], coords[:, 0], '+r', markersize = 15)
    plt.axis('off')   

show_image_with_detected_corners(image, coords)







# --------------------------------------------------------------------------------------------------------- #

from skimage.io import imread
from skimage import color

building_image = imread('Corners_Building_Top.jpg')

##                  Perspective                  ##
# Import the corner detector related functions and module
from skimage.feature import corner_harris, corner_peaks

# Convert image from RGB-3 to grayscale
building_image_gray = color.rgb2gray(building_image)

# Apply the detector  to measure the possible corners
measure_image = corner_harris(building_image_gray)

# Find the peaks of the corners using the Harris detector
coords = corner_peaks(corner_harris(building_image_gray), min_distance = 2)

# Show original and resulting image with corners detected
show_image(building_image, "Original")
show_image_with_corners(building_image, coords)





##                  Less corners                  ##
# Part 1
# Find the peaks with a min distance of 2 pixels
coords_w_min_2 = corner_peaks(measure_image, min_distance = 2)
print("With a min_distance set to 2, we detect a total", len(coords_w_min_2), "corners in the image.")
# output: With a min_distance set to 2, we detect a total 98 corners in the image.


# Part 2
# Find the peaks with a min distance of 40 pixels
coords_w_min_40 = corner_peaks(measure_image, min_distance = 40)
print("With a min_distance set to 40, we detect a total", len(coords_w_min_40), "corners in the image.")
# output: With a min_distance set to 40, we detect a total 36 corners in the image.


# Part 3
# Show original and resulting image with corners detected
show_image_with_corners(building_image, coords_w_min_2, "Corners detected with 2 px of min_distance")
show_image_with_corners(building_image, coords_w_min_40, "Corners detected with 40 px of min_distance")