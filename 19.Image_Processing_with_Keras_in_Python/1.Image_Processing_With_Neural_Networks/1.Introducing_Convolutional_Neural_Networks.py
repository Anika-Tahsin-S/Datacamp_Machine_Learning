# Import matplotlib
import matplotlib.pyplot as plt

data = plt.imread('stop_sign.jpg')
plt.imshow(data)
plt.show()

data.shape
(2832, 4256, 3)
# The first two dimensions correspond to the height and width of the image (the number of pixels). 
# The last dimension corresponds to the red, green and blue colors present in each pixel. 


# Modifying image data
data[:, :, 1] = 0
data[:, :, 2] = 0

plt.imshow(data)
plt.show()


# Changing an image
data[200:1200, 200:1200, :] = [0, 1, 0]
plt.imshow(data)

# Black and white images
tshirt[10:20, 15:25] = 1
plt.imshow(data)
plt.show()





import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------------------------- #
##                  Images as data: visualizations                  ##
# Import matplotlib
import matplotlib.pyplot as plt

# Load the image
data = plt.imread('bricks.png')

# Display the image
plt.imshow(data)
plt.show()


##                  Images as data: changing images                  ##
# Set the red channel in this part of the image to 1
data[:10, :10, 0] = 1

# Set the green channel in this part of the image to 0
data[:10, :10, 1] = 0

# Set the blue channel in this part of the image to 0
data[:10, :10, 2] = 0

# Visualize the result
plt.imshow(data)
plt.show()
