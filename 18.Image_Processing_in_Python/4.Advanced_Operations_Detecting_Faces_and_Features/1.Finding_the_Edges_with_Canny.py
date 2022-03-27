import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread



## Edge Detection
from skimage.feature import canny

coins = color.rgb2gray(coins)
canny_edges = canny(coins)
show_image(canny_edges, "Edge with Canny")

# Canny edge detector
canny_edges_0_5 = canny(coins, sigma = 0.5)
show_image(canny_edges, "Sigma with 1")
show_image(canny_edges_0_5, "Sigma with 0.4")





def show_image(image, title='Image', cmap_type='gray'):
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.axis('off')
    
def plot_comparison(img_original, img_filtered, img_title_filtered):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 8), sharex=True, sharey=True)
    ax1.imshow(img_original, cmap=plt.cm.gray)
    ax1.set_title('Original')
    ax1.axis('off')
    ax2.imshow(img_filtered, cmap=plt.cm.gray)
    ax2.set_title(img_title_filtered)
    ax2.axis('off')



# --------------------------------------------------------------------------------------------------------- #
from skimage import color

grapefruit = imread('Grape_Fruit.jpg')

##                  Edges                  ##
# Import the canny edge detector 
from skimage.feature import canny

# Convert image to grayscale
grapefruit = color.rgb2gray(grapefruit)

# Apply canny edge detector
canny_edges = canny(grapefruit)

# Show resulting image
show_image(canny_edges, "Edges with Canny")





##                  Less Edgy                  ##
# Part 1
# Apply canny edge detector with a sigma of 1.8
canny_edges = canny(grapefruit, sigma = 1.8)

# Part 2
# Apply canny edge detector with a sigma of 2.2
edges_2_2 = canny(grapefruit, 2.2)

# Part 3
# Show resulting images
show_image(edges_1_8, "Sigma of 1.8")
show_image(edges_2_2, "Sigma of 2.2")