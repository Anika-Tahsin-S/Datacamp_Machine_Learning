# Selecting layers
model.layers

# Getting model weights
conv1 = model.layers[0]
weights1 = conv1.get_weights()
len(weights1)

kernels1 = weights1[0]
kernels1.shape
(3, 3, 1, 5)
# The first item in this list is an array that holds the values of the weights for the convolutional kernels for this layer. The kernels array has the shape 3 by 3 by 1 by 5. 
# The first 2 dimensions denote the kernel size. This network was initialized with kernel size of 3. 
# The third dimension denotes the number of channels in the kernels. This is one, because the network was looking at black and white data. 
# The last dimension denotes the number of kernels in this layer: 5.

kernels1_1 = kernels1[:, :, 0, 0]
kernels1_1.shape


# Visualizing the kernel
plt.imshow(kernels1_1)


# Visualizing the kernel responses
test_image = test_data[3, :, :, 0]
plt.imshow(test_image)
filtered_image = convolution(test_image, kernels1_1)
plt.imshow(filtered_image)

test_image = test_data[4, :, :, 1]
plt.imshow(test_image)
filtered_img = convolution(test_image, kernels1_1)
plt.imshow(filtered_img)


kernel1_2 = test_data[:, :, 0, 1]
filtered_img = convolution(test_image, kernel1_2)
plt.imshow(filtered_img)





from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D

model = Sequential()

model.add(Conv2D(5, kernel_size = 2, activation = 'relu', input_shape = (img_rows, img_cols, 1)))
model.add(Conv2D(15, kernel_size = 2, activation = 'relu'))
model.add(MaxPool2D(2))
model.add(Flatten())
model.add(Dense(3, activation = 'softmax'))

model.summary()
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

checkpoint = ModelCheckpoint('weights_fasion.hdf5', monitor='val_loss', save_best_only = True)

model.fit(train_data, train_labels, epochs = 3, validation_split = 0.2, batch_size = 10,
          callbacks = [checkpoint])



# --------------------------------------------------------------------------------------------------------- #
##                  Extracting a kernel from a trained network                  ##
# Load the weights into the model
model.load_weights('weights.hdf5')

# Get the first convolutional layer from the model
c1 = model.layers[0]

# Get the weights of the first convolutional layer
weights1 = c1.get_weights()

# Pull out the first channel of the first kernel in the first layer
kernel = weights1[0][:, :,0, 0]
print(kernel)
# output:
#     [[ 0.03504268  0.4328133 ]
#      [-0.17416623  0.4680562 ]]






##                  Shape of the weights                  ##
# A Keras neural network stores its layers in a list called model.layers. For the convolutional layers, you can get the weights using the .get_weights() method. This returns a list, and the first item in this list is an array representing the weights of the convolutional kernels. 
# If the shape of this array is (2, 2, 1, 5), what does the first number (2) represent?

# Load the weights from file
model.load_weights('weights.hdf5')

# Predict from the first three images in the test data
model.predict(test_data)

model.layers
# output:
# [<keras.layers.convolutional.Conv2D at 0x7f852c21b070>,
#  <keras.layers.convolutional.Conv2D at 0x7f852c21b4f0>,
#  <keras.layers.pooling.MaxPooling2D at 0x7f8445315550>,
#  <keras.layers.core.Flatten at 0x7f852c26e130>,
#  <keras.layers.core.Dense at 0x7f84453155b0>]

# ANswer: The kernel size is 2 by 2.
# Each of the 2s in this shape is one of the dimensions of the kernel.





##                  Visualizing kernel responses                  ##
import numpy as np
import pandas as pd

def convolution(image, kernel):
    kernel = kernel - kernel.mean()
    result = np.zeros(image.shape)

    for ii in range(image.shape[0]-2):
        for jj in range(image.shape[1]-2):
            result[ii, jj] = np.sum(image[ii:ii+2, jj:jj+2] * kernel)

    return result

import matplotlib.pyplot as plt

# Convolve with the fourth image in test_data
out = convolution(test_data[3, :, :, 0], kernel)

# Visualize the result
plt.imshow(out)
plt.show()

