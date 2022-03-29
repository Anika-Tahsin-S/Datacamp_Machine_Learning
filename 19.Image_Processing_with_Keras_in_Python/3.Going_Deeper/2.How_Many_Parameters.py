# Counting Parameters
from keras.models import Sequential
from keras.layers import Dense, Flatten

model = Sequential()

model.add(Dense(15, kernel_size = 2, activation = 'relu', input_shape = (784,)))
# parameters = 784 * 10 + 10 = 7850
model.add(Dense(10, activation = 'relu'))
# parameters = 10 * 10 + 10 = 110
model.add(Dense(3, activation = 'softmax'))
# parameters = 3 * 10 + 10 = 33
# Params = 7993
model.summary()


# The number of parameters in a CNN
from keras.layers import Dense, Conv2D, Flatten

model = Sequential()

model.add(Conv2D(10, kernel_size = 3, activation = 'relu', input_shape = (28, 28, 1), padding = 'same'))
# parameters = (3 * 3) * 10 + 10 = 100
model.add(Conv2D(5, kernel_size = 3, activation = 'relu', padding = 'same'))
# parameters = 10 *  9 * 10 + 10 = 910

model.add(Flatten())
# parameters = 0

model.add(Dense(3, activation = 'softmax'))
# parameters = 7840 * 3 + 3 = 23523
# Params = 100 + 910 + 0 + 23523 = 24533

model.summary()


# Increasing the number of units in each layer
model = Sequential()

model.add(Conv2D(5, kernel_size = 3, activation = 'relu', input_shape = (28, 28, 1), padding = 'same'))
model.add(Conv2D(15, kernel_size = 3, activation = 'relu', padding = 'same'))

model.add(Flatten())
model.add(Dense(3, activation = 'softmax'))
# Params = 36023

model.summary()








# --------------------------------------------------------------------------------------------------------- #
##                  How many parameters in a CNN?                  ##
# We need to know how many parameters a CNN has, so we can adjust the model architecture, to reduce this number or shift parameters from one part of the network to another. How many parameters would a network have if its inputs are images with 28-by-28 pixels, there is one convolutional layer with 10 units kernels of 3-by-3 pixels, using zero padding (input has the same size as the output), and one densely connected layer with 2 units?

10 * (3 * 3) + 10 + 784 * 10 * 2 + 2 = 15,782
# Answer: 15,782



##                  How many parameters in a deep CNN?                  ##
# CNN model
model = Sequential()
model.add(Conv2D(10, kernel_size = 2, activation = 'relu', 
                 input_shape = (28, 28, 1)))
model.add(Conv2D(10, kernel_size=  2, activation = 'relu'))
model.add(Flatten())
model.add(Dense(3, activation = 'softmax'))

# Summarize the model 
model.summary()

# output:
    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 27, 27, 10)        50        
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 26, 26, 10)        410       
    _________________________________________________________________
    flatten (Flatten)            (None, 6760)              0         
    _________________________________________________________________
    dense (Dense)                (None, 3)                 20283     
    =================================================================
    Total params: 20,743
    Trainable params: 20,743
    Non-trainable params: 0
    _________________________________________________________________
