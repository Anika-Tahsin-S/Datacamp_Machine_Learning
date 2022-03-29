# Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout

model = Sequential()

# Add a convolutional layer
model.add(Conv2D(5, kernel_size = 3, activation = 'relu', input_shape = (img_rows, img_cols, 1)))

model.add(Dropout(0.25))
model.add(Conv2D(15, kernel_size = 3, activation = 'relu'))
model.add(Flatten())
model.add(Dense(3, activation = 'softmax'))


# Batch Normalization
# This operation takes the output of a particular layer, and rescales it so that it always has 0 mean and standard deviation of 1 in every batch of training.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, BatchNormalization

model = Sequential()

# Add a convolutional layer
model.add(Conv2D(5, kernel_size = 3, activation = 'relu', input_shape = (img_rows, img_cols, 1)))

model.add(BatchNormalization())
model.add(Conv2D(15, kernel_size = 3, activation = 'relu'))
model.add(Flatten())
model.add(Dense(3, activation = 'softmax'))


# Using them together carefully
# Sometimes dropout and batch normalization do not work well together. 
# This is because while dropout slows down learning, making it more incremental and careful, batch normalization tends to make learning go faster. 
# Their effects together may in fact counter each other, and networks sometimes perform worse when both of these methods are used together than they would if neither were used. 
# This has been called "the disharmony of batch normalization and dropout". 








# --------------------------------------------------------------------------------------------------------- #
##                  Adding dropout to your network                  ##
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout

# Add a convolutional layer
model.add(Conv2D(15, kernel_size = 2, activation = 'relu', 
                 input_shape = (img_rows, img_cols, 1)))

# Add a dropout layer
model.add(Dropout(0.2))

# Add another convolutional layer
model.add(Conv2D(5, kernel_size = 2, activation = 'relu'))

# Flatten and feed to output layer
model.add(Flatten())
model.add(Dense(3, activation = 'softmax'))
model.summary()

# output:
    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 27, 27, 15)        75        
    _________________________________________________________________
    dropout (Dropout)            (None, 27, 27, 15)        0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 26, 26, 5)         305       
    _________________________________________________________________
    flatten (Flatten)            (None, 3380)              0         
    _________________________________________________________________
    dense (Dense)                (None, 3)                 10143     
    =================================================================
    Total params: 10,523
    Trainable params: 10,523
    Non-trainable params: 0
    _________________________________________________________________







##                  Add batch normalization to your network                  ##
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, BatchNormalization

# Add a convolutional layer
model.add(Conv2D(15, kernel_size = 2, activation = 'relu', input_shape = (img_rows, img_cols, 1)))


# Add batch normalization layer
model.add(BatchNormalization())

# Add another convolutional layer
model.add(Conv2D(5, kernel_size = 2, activation = 'relu'))

# Flatten and feed to output layer
model.add(Flatten())
model.add(Dense(3, activation = 'softmax'))
model.summary()

# output:
    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 27, 27, 15)        75        
    _________________________________________________________________
    batch_normalization (BatchNo (None, 27, 27, 15)        60        
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 26, 26, 5)         305       
    _________________________________________________________________
    flatten (Flatten)            (None, 3380)              0         
    _________________________________________________________________
    dense (Dense)                (None, 3)                 10143     
    =================================================================
    Total params: 10,583
    Trainable params: 10,553
    Non-trainable params: 30
    _________________________________________________________________

