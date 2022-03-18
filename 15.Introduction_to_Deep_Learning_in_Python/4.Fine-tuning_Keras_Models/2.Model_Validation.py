# Early Stopping
from gc import callbacks
from keras.callbacks import EarlyStopping

early_stopping_monitor = EarlyStopping(patience = 2)

model.fit(pred, target, val_split = 0.3, nb_epoch = 20, callbacks = [early_stopping_monitor])





# --------------------------------------------------------------------------------------------------------- #
##                   Evaluating Model Accuracy on Validation Dataset                  ##
# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]
input_shape = (n_cols,)

# Specify the model
model = Sequential()
model.add(Dense(100, activation = 'relu', input_shape = input_shape))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(2, activation = 'softmax'))

# Compile the model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fit the model
hist = model.fit(predictors, target, validation_split = 0.3)

# output:
#     Train on 623 samples, validate on 268 samples
#     Epoch 1/10
    
#  32/623 [>.............................] - ETA: 27s - loss: 3.3028 - acc: 0.4062608/623 
#         [============================>.] - ETA: 0s - loss: 1.2774 - acc: 0.5938623/623 
#         [==============================] - 1s - loss: 1.2568 - acc: 0.6003 - val_loss: 0.5920 - val_acc: 0.7201
# ............................................
#     Epoch 10/10
#     
#  32/623 [>.............................] - ETA: 0s - loss: 0.5042 - acc: 0.7500512/623 
#         [=======================>......] - ETA: 0s - loss: 0.6283 - acc: 0.6973623/623 
#         [==============================] - 0s - loss: 0.6190 - acc: 0.6998 - val_loss: 0.5360 - val_acc: 0.7500








##                   Early stopping: Optimizing the optimization                  ##
# Import EarlyStopping
from keras.callbacks import EarlyStopping

# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]
input_shape = (n_cols,)

# Specify the model
model = Sequential()
model.add(Dense(100, activation='relu', input_shape = input_shape))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Define early_stopping_monitor
early_stopping_monitor = EarlyStopping(patience = 2)

# Fit the model
model.fit(predictors, target, validation_split = 0.3, epochs = 30, callbacks = [early_stopping_monitor])

# output:
#     Train on 623 samples, validate on 268 samples
#     Epoch 1/30
    
#  32/623 [>.............................] - ETA: 38s - loss: 5.6563 - acc: 0.4688416/623 
#         [===================>..........] - ETA: 1s - loss: 1.7578 - acc: 0.5216623/623 
#         [==============================] - 2s - loss: 1.6355 - acc: 0.5650 - val_loss: 1.0815 - val_acc: 0.6642

# ..........................................................................
#     Epoch 6/30
    
#  32/623 [>.............................] - ETA: 0s - loss: 0.4535 - acc: 0.8438576/623 
#         [==========================>...] - ETA: 0s - loss: 0.6293 - acc: 0.7135623/623 
#         [==============================] - 0s - loss: 0.6267 - acc: 0.7095 - val_loss: 0.5812 - val_acc: 0.7015
#     Epoch 7/30
    
#  32/623 [>.............................] - ETA: 0s - loss: 0.6521 - acc: 0.6562512/623 
#         [=======================>......] - ETA: 0s - loss: 0.6306 - acc: 0.7031623/623 
#         [==============================] - 0s - loss: 0.6556 - acc: 0.6998 - val_loss: 0.6640 - val_acc: 0.6716

# Because optimization will automatically stop when it is no longer helpful, it is okay to specify the maximum number of epochs as 30 rather than using the default of 10 that you've used so far. Here, it seems like the optimization stopped after 7 epochs.








##                   Experimenting with wider networks                  ##
# IPython Shell
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 10)                110       
_________________________________________________________________
dense_2 (Dense)              (None, 10)                110       
_________________________________________________________________
dense_3 (Dense)              (None, 2)                 22        
=================================================================
Total params: 242.0
Trainable params: 242
Non-trainable params: 0.0
_________________________________________________________________
None
# ............................................... #

# Define early_stopping_monitor
early_stopping_monitor = EarlyStopping(patience = 2)

# Create the new model: model_2
model_2 = Sequential()

# Add the first and second layers
model_2.add(Dense(100, activation = 'relu', input_shape = input_shape))
model_2.add(Dense(100, activation = 'relu'))

# Add the output layer
model_2.add(Dense(2, activation='softmax'))

# Compile model_2
model_2.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fit model_1
model_1_training = model_1.fit(predictors, target, epochs = 15, validation_split = 0.2, callbacks = [early_stopping_monitor], verbose = False)

# Fit model_2
model_2_training = model_2.fit(predictors, target, epochs = 15, validation_split = 0.2, callbacks = [early_stopping_monitor], verbose = False)

# Create the plot
plt.plot(model_1_training.history['val_loss'], 'r', model_2_training.history['val_loss'], 'b')
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.show()







##                   Adding layers to a network                  ##

# IPython Shell
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 50)                550       
_________________________________________________________________
dense_2 (Dense)              (None, 2)                 102       
=================================================================
Total params: 652.0
Trainable params: 652
Non-trainable params: 0.0
_________________________________________________________________
None
# ............................................... #



# The input shape to use in the first hidden layer
input_shape = (n_cols,)

# Create the new model: model_2
model_2 = Sequential()

# Add the first, second, and third hidden layers
model_2.add(Dense(50, activation = 'relu', input_shape = input_shape))
model_2.add(Dense(50, activation = 'relu'))
model_2.add(Dense(50, activation = 'relu'))

# Add the output layer
model_2.add(Dense(2, activation='softmax'))

# Compile model_2
model_2.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fit model 1
model_1_training = model_1.fit(predictors, target, epochs = 20, validation_split = 0.4, callbacks = [early_stopping_monitor], verbose = False)

# Fit model 2
model_2_training = model_2.fit(predictors, target, epochs=20, validation_split = 0.4, callbacks = [early_stopping_monitor], verbose = False)

# Create the plot
plt.plot(model_1_training.history['val_loss'], 'r', model_2_training.history['val_loss'], 'b')
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.show()