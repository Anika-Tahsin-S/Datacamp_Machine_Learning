## Unsupervised Segmentation
# Simple Linear Iterative Clustering (SLIC)

from skimage.segmentation import slic
from skimage.color import label2rgb

segments = slic(image)

segmented_image = label2rgb(segments, image, kind = 'avg')

show_image(image)
show_image(segmented_image, "Segmented Image")

# More Segments
segments = slic(image, n_segments = 300)
segmented_image = label2rgb(segments, image, kind = 'avg')

show_image(segmented_image)












# --------------------------------------------------------------------------------------------------------- #
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import io
face_image = io.imread('Face_Image.jpg')

def show_image(image, title='Image', cmap_type='gray'):
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.axis('off')


##                  Number of pixels                  ##
face_image.shape
# Output: (265, 191, 3)
# Answer: face_image is 265 * 191 = 50,615 pixels


##                  Superpixel segmentation                  ##
# Import the slic function from segmentation module
from skimage.segmentation import slic

# Import the label2rgb function from color module
from skimage.color import label2rgb

# Obtain the segmentation with 400 regions
segments = slic(face_image, n_segments =  400)

# Put segments on top of original image to compare
segmented_image = label2rgb(segments, face_image, kind = 'avg')

# Show the segmented image
show_image(segmented_image, "Segmented image, 400 superpixels")