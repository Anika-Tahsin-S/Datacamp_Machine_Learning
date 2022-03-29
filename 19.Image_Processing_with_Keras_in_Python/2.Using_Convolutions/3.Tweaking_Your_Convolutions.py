# Zero padding in keras
model.add(Conv2D(10, kernel_size = 3, activation = 'relu', input_shape = (img_rows, img_cols, 1)), padding = 'same')

# Strides in keras
model.add(Conv2D(10, kernel_size = 3, activation = 'relu', input_shape = (img_rows, img_cols, 1)), strides = 1)
# if the strides is more than 1 then the kernel jumps in steps of that number of pixels. This also means that the output will be smaller. 
model.add(Conv2D(10, kernel_size = 3, activation = 'relu', input_shape = (img_rows, img_cols, 1)), strides = 2)


# Calculating the size of the output
O = ((I - K + 2 * P)/S) + 1
# Here, I = size of the input
# K = size of the kernal
# P = size of the zero padding
# S = strides

# Dilation in Keras
model.add(Conv2D(10, kernel_size = 3, activation = 'relu', input_shape = (img_rows, img_cols, 1)), dilation_rate = 2)







# --------------------------------------------------------------------------------------------------------- #
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense

##                  Add padding to a CNN                  ##
# Initialize the model
model = Sequential()

# Add the convolutional layer
model.add(Conv2D(10, kernel_size = 3, activation = 'relu', 
                 input_shape=(img_rows, img_cols, 1), 
                 padding = 'same'))

# Feed into output layer
model.add(Flatten())
model.add(Dense(3, activation = 'softmax'))





##                  Add strides to a convolutional network                  ##
# Initialize the model
model = Sequential()

# Add the convolutional layer
model.add(Conv2D(10, kernel_size = 3, activation = 'relu', 
              input_shape = (img_rows, img_cols, 1), 
              strides = 2))

# Feed into output layer
model.add(Flatten())
model.add(Dense(3, activation = 'softmax'))
# With strides set to 2, the network skips every other pixel.



##                  Calculate the size of convolutional layer output                  ##
# Zero padding and strides affect the size of the output of a convolution. 
# What is the size of the output for an input of size 256 by 256, with a kernel of size 4 by 4, padding of 1 and strides of 2?

# Answer: 128
# O = ((I - K + 2 * P)/S) + 1
# O = ((256 - 4 + 2 * 1)/2) + 1 = 128
