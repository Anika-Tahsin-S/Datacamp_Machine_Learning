## Shapes in scikit-image
from skimage import morphology

square = morphology.square(4)
rectangle = morphology.rectangle(4, 2)


## Erosion in scikit-image
from skimage import morphology

selem = rectangle(12, 6)
eroded_image = morphology.binary_erosion(image_horse, selem = selem)
plot_comparison(image_horse, eroded_image, 'Erosion')



## Binary erosion with default selem
eroded_image = morphology.binary_erosion(image_horse)


## Dilation in scikit-image
from skimage import morphology

selem = rectangle(12, 6)
dilated_image = morphology.binary_dilation(image_horse)
plot_comparison(image_horse, dilated_image, 'Erosion')









# --------------------------------------------------------------------------------------------------------- #
##                  Handwritten letters                  ##
from skimage import io
upper_r_image = io.imread('Upper_R_Image.jpg')
# Import the morphology module
from skimage import morphology

# Obtain the eroded shape 
eroded_image_shape = morphology.binary_erosion(upper_r_image) 

# See results
show_image(upper_r_image, 'Original')
show_image(eroded_image_shape, 'Eroded image')






##                  Improving thresholded image                  ##
world_image = io.imread('World_Image.jpg')

# Import the module
from skimage import morphology

# Obtain the dilated image 
dilated_image = morphology.binary_dilation(world_image)

# See results
show_image(world_image, 'Original')
show_image(dilated_image, 'Dilated image')
