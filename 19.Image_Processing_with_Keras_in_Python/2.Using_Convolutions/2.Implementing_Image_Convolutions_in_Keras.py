# Keras Convolution Layer
from keras.layers import Conv2D
Conv2D(10, kernal_size = 3, activation = 'relu')

# Integrating convolution layers into a network
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(10, kernel_size = 3, activation = 'relu', input_shape = (img_rows, img_cols, 1)))
model.add(Flatten())
model.add(Dense(3, activation = 'softmax'))

model.compile(optimizer = "adam", loss ='categorical+crossentropy', metrics = ['accuracy'])

model.fit(train_data, train_labels, validation_split = 0.2, epochs = 3)

model.evaluate(test_data, test_labels, epochs = 3)









# --------------------------------------------------------------------------------------------------------- #
##                  Convolutional network for image classification                  ##
kernel = np.array([[-1, -1, -1], 
          [1, 1, 1], 
          [-1, -1, -1]])

# Import the necessary components from Keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

# Initialize the model object
model = Sequential()

# Add a convolutional layer
model.add(Conv2D(10, kernel_size = 3, activation = 'relu', 
               input_shape = (img_rows, img_cols, 1)))

# Flatten the output of the convolutional layer
model.add(Flatten())
# Add an output layer for the 3 categories
model.add(Dense(3, activation = 'softmax'))





##                  Training a CNN to classify clothing types                  ##
# Compile the model 
model.compile(optimizer = 'adam', 
              loss = 'categorical_crossentropy', 
              metrics = ['accuracy'])

# Fit the model on a training set
model.fit(train_data, train_labels, 
          validation_split = 0.2, 
          epochs = 3, batch_size = 10)

# output:
#     Epoch 1/3
#     
# 1/4 [======>.......................] - ETA: 2s - loss: 1.0927 - accuracy: 0.30004/4 [==============================] - 1s 155ms/step - loss: 0.8978 - accuracy: 0.5250 - val_loss: 0.5467 - val_accuracy: 1.0000
#     Epoch 2/3
#     
# 1/4 [======>.......................] - ETA: 0s - loss: 0.5905 - accuracy: 1.00004/4 [==============================] - 0s 10ms/step - loss: 0.5098 - accuracy: 0.9250 - val_loss: 0.3532 - val_accuracy: 1.0000
#     Epoch 3/3
#     
# 1/4 [======>.......................] - ETA: 0s - loss: 0.3706 - accuracy: 0.90004/4 [==============================] - 0s 9ms/step - loss: 0.3059 - accuracy: 0.9500 - val_loss: 0.2645 - val_accuracy: 1.0000




##                  Evaluating a CNN with test data                  ##
# Evaluate the model on separate test data
model.evaluate(test_data, test_labels, batch_size = 10)

# output:   
# 1/1 [==============================] - ETA: 0s - loss: 0.2906 - accuracy: 0.90001/1 [==============================] - 0s 22ms/step - loss: 0.2906 - accuracy: 0.9000
