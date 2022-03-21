# The Architecture
from keras.models import Sequential
from keras.layers import Dense


model = Sequential()
# Add an input and dense layer
model.add(Dense(2, input_shape = (1,)))
# Add an output layer for 3 classes and sigmoid activation
model.add(Dense(3, activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy')

model.fit(X_train, y_train, epochs = 100, validation_split = 0.2)







# --------------------------------------------------------------------------------------------------------- #
##                   An irrigation machine                  ##
# Instantiate a Sequential model
model = Sequential()

# Add a hidden layer of 64 neurons and a 20 neuron's input
model.add(Dense(64, input_shape = (20,), activation = 'relu'))

# Add an output layer of 3 neurons with sigmoid activation
model.add(Dense(3, activation = 'sigmoid'))

# Compile your model with binary crossentropy loss
model.compile(optimizer = 'adam',
           loss = 'binary_crossentropy',
           metrics = ['accuracy'])

model.summary()

# output:
#     Model: "sequential_1"
#     _________________________________________________________________
#     Layer (type)                 Output Shape              Param #   
#     =================================================================
#     dense_1 (Dense)              (None, 64)                1344      
#     _________________________________________________________________
#     dense_2 (Dense)              (None, 3)                 195       
#     =================================================================
#     Total params: 1,539
#     Trainable params: 1,539
#     Non-trainable params: 0
#     _________________________________________________________________








##                   Training with multiple labels                  ##
# Train for 100 epochs using a validation split of 0.2
model.fit(sensors_train, parcels_train, epochs = 100, validation_split = 0.2)

# Predict on sensors_test and round up the predictions
preds = model.predict(sensors_test)
preds_rounded = np.round(preds)

# Print rounded preds
print('Rounded Predictions: \n', preds_rounded)

# Evaluate your model's accuracy on the test data
accuracy = model.evaluate(sensors_test, parcels_test)[1]

# Print accuracy
print('Accuracy:', accuracy)

# output:
#     Train on 1120 samples, validate on 280 samples
#     Epoch 1/100
#     
#       32/1120 [..............................] - ETA: 9s - loss: 1.2917 - acc: 0.3750
#      864/1120 [======================>.......] - ETA: 0s - loss: 0.7829 - acc: 0.5818
#     1120/1120 [==============================] - 0s 326us/step - loss: 0.7333 - acc: 0.6164 - val_loss: 0.5356 - val_acc: 0.7369
# 
#     Epoch 100/100
#     
#       32/1120 [..............................] - ETA: 0s - loss: 0.1878 - acc: 0.9271
#      768/1120 [===================>..........] - ETA: 0s - loss: 0.1261 - acc: 0.9466
#     1120/1120 [==============================] - 0s 86us/step - loss: 0.1249 - acc: 0.9476 - val_loss: 0.2776 - val_acc: 0.8833
#     Rounded Predictions: 
#      [[1. 1. 0.]
#      [0. 1. 0.]
#      [0. 1. 0.]
#      ...
#      [1. 1. 0.]
#      [0. 1. 0.]
#      [1. 1. 1.]]
    
#     32/600 [>.............................] - ETA: 0s
#     600/600 [==============================] - 0s 34us/step
#     Accuracy: 0.9061111267407735