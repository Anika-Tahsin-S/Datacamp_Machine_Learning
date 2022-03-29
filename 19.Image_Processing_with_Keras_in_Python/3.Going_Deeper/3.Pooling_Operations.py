# Implementing max pooling
result = np.zeros((imt.shape[0] // 2, imt.shape[1] // 2))
result[0, 0] = np.max(im[0:2, 0:2])
result[0, 1] = np.max(im[0:2, 2:4])
result[0, 2] = np.max(im[0:2, 4:6])
........
result[1, 0] = np.max(im[2:4, 0:2])
result[1, 1] = np.max(im[2:4, 2:4])
...
# Another way to implement this operation
# Pooling operation
for ii in range(result.shape[0]):
    for jj in range(result.shape[1]):
        result[ii, jj] = np.max(imt[ii * 2:ii * 2 + 2, jj * 2:jj * 2 + 2])


from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D

model = Sequential()

model.add(Dense(5, kernel_size = 3, activation = 'relu', input_shape = (img_rows, img_cols, 1)))
# Add a pooling operation
model.add(MaxPool2D(2))

model.add(Dense(15, kernel_size = 3, activation = 'relu', input_shape = (img_rows, img_cols, 1)))
model.add(MaxPool2D(2))

model.add(Flatten())
model.add(Dense(3, activation = 'softmax'))
model.summary()







# --------------------------------------------------------------------------------------------------------- #
##                  Write your own pooling operation                  ##
# Result placeholder
result = np.zeros((im.shape[0]//2, im.shape[1]//2))

# Pooling operation
for ii in range(result.shape[0]):
    for jj in range(result.shape[1]):
        result[ii, jj] = np.max(im[ii * 2:ii * 2 + 2, jj * 2:jj * 2 + 2])






##                  Keras pooling layers                  ##
# Add a convolutional layer
model.add(Conv2D(15, kernel_size = 2, activation = 'relu', 
                 input_shape = (img_rows, img_cols, 1)))

# Add a pooling operation
model.add(MaxPool2D(2))

# Add another convolutional layer
model.add(Conv2D(5, kernel_size = 2, activation = 'relu', 
                 input_shape = (img_rows, img_cols, 1)))

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
    max_pooling2d (MaxPooling2D) (None, 13, 13, 15)        0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 12, 12, 5)         305       
    _________________________________________________________________
    flatten (Flatten)            (None, 720)               0         
    _________________________________________________________________
    dense (Dense)                (None, 3)                 2163      
    =================================================================
    Total params: 2,543
    Trainable params: 2,543
    Non-trainable params: 0
    _________________________________________________________________






##                  Train a deep CNN with pooling to classify images                  ##
# Compile the model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

# Fit to training data
model.fit(train_data, train_labels, epochs = 3, validation_split = 0.2, batch_size=10)

# Evaluate on test data
model.evaluate(test_data, test_labels, batch_size = 10)


# output:
#     Epoch 1/3
#     
# 1/4 [======>.......................] - ETA: 4s - loss: 1.0943 - accuracy: 0.20004/4 [==============================] - 2s 256ms/step - loss: 1.0903 - accuracy: 0.2250 - val_loss: 0.9586 - val_accuracy: 0.8000
#     Epoch 2/3
#     
# 1/4 [======>.......................] - ETA: 0s - loss: 1.0089 - accuracy: 0.50004/4 [==============================] - 0s 14ms/step - loss: 0.9784 - accuracy: 0.5250 - val_loss: 0.8559 - val_accuracy: 0.9000
#     Epoch 3/3
#     
# 1/4 [======>.......................] - ETA: 0s - loss: 0.9119 - accuracy: 0.60004/4 [==============================] - 0s 12ms/step - loss: 0.8877 - accuracy: 0.6250 - val_loss: 0.7540 - val_accuracy: 1.0000
#     
# 1/1 [==============================] - ETA: 0s - loss: 0.9465 - accuracy: 0.60001/1 [==============================] - 0s 20ms/step - loss: 0.9465 - accuracy: 0.6000

