# Network with one convolutional layer: implementation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

model = Sequential()
model.add(Conv2D(10, kernel_size = 2, activation = 'relu', 
                 input_shape=(img_rows, img_cols, 1)))
model.add(Flatten())
model.add(Dense(3, activation = 'softmax'))

# Building a deeper network
model = Sequential()
model.add(Conv2D(10, kernel_size = 2, activation = 'relu', 
                 input_shape=(img_rows, img_cols, 1), padding = 'equal'))

model.add(Conv2D(10, kernel_size = 2, activation = 'relu')
model.add(Flatten())
model.add(Dense(3, activation = 'softmax'))






# --------------------------------------------------------------------------------------------------------- #
##                  Creating a deep learning network                  ##
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

img_cols, img_rows = 28, 28

model = Sequential()

# Add a convolutional layer (15 units)
model.add(Conv2D(15, kernel_size = 2, activation = 'relu', input_shape = (img_rows, img_cols, 1)))


# Add another convolutional layer (5 units)
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






##                  Train a deep CNN to classify clothing images                  ##
# Compile model
model.compile(optimizer = 'adam', 
              loss = 'categorical_crossentropy', 
              metrics = ['accuracy'])

# Fit the model to training data 
model.fit(train_data, train_labels, 
          validation_split = 0.2, 
          epochs = 3, batch_size = 10)

# Evaluate the model on test data
model.evaluate(test_data, test_labels, batch_size = 10)

# output:
#     Epoch 1/3
#     
# 1/4 [======>.......................] - ETA: 5s - loss: 1.1121 - accuracy: 0.20004/4 [==============================] - 3s 250ms/step - loss: 0.9827 - accuracy: 0.5250 - val_loss: 0.7748 - val_accuracy: 0.8000
#     Epoch 2/3
#     
# 1/4 [======>.......................] - ETA: 0s - loss: 0.7910 - accuracy: 0.70004/4 [==============================] - 0s 13ms/step - loss: 0.6839 - accuracy: 0.9000 - val_loss: 0.4973 - val_accuracy: 1.0000
#     Epoch 3/3
    
# 1/4 [======>.......................] - ETA: 0s - loss: 0.5136 - accuracy: 1.00004/4 [==============================] - 0s 19ms/step - loss: 0.4825 - accuracy: 0.9250 - val_loss: 0.3594 - val_accuracy: 1.0000
#     
# 1/1 [==============================] - ETA: 0s - loss: 0.4442 - accuracy: 0.90001/1 [==============================] - 0s 30ms/step - loss: 0.4442 - accuracy: 0.9000





##                  What is special about a deep network?                  ##
# Networks with more convolution layers are called "deep" networks, and they may have more power to fit complex data, because of their ability to create hierarchical representations of the data that they fit.
# What is a major difference between a deep CNN and a CNN with only one convolutional layer?

# Answer: A deep network requires more data and more computation to fit.
