## History Callback ##
# Training a model and saving its history
history = model.fit(X_train, y_train, epochs = 100, metrics = ['accuracy'])
print(history.history['loss'])
print(history.history['acc'])

# Validation
history = model.fit(X_train, y_train, epochs = 100, 
                    validation_data = (X_test, y_test),
                    metrics = ['accuracy'])
print(history.history['val_loss'])
print(history.history['val_acc'])

# History plots
plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
# Make it Pretty
plt.title("Model Accuracy")
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'])
plt.show()



## Early Stopping ##
from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor = 'val_loss', patience = 5)

model.fit(X_train, y_train, epochs = 100, validation_data = (X_test, y_test), callbacks = [early_stopping])



## Model Checkpoint ##
from keras.callbacks import ModelCheckpoint

model_save = ModelCheckpoint('best_model.hdf5', save_best_only = True)

model.fit(X_train, y_train, epochs = 100, validation_data = (X_test, y_test), callbacks = [model_save])








# --------------------------------------------------------------------------------------------------------- #
##                   The history callback                  ##
# Train your model and save its history
h_callback = model.fit(X_train, y_train, epochs = 50,
               validation_data = (X_test, y_test))

# Plot train vs test loss during training
plot_loss(h_callback.history['loss'], h_callback.history['val_loss'])

# Plot train vs test accuracy during training
plot_accuracy(h_callback.history['acc'], h_callback.history['val_acc'])





##                   Early stopping your model                  ##
# Import the early stopping callback
from keras.callbacks import EarlyStopping

# Define a callback to monitor val_acc
monitor_val_acc = EarlyStopping(monitor = 'val_acc', patience = 5)

# Train your model using the early stopping callback
model.fit(X_train, y_train, epochs = 1000, validation_data = (X_test, y_test), callbacks = [monitor_val_acc])






##                   A Combination of Callbacks                  ##
# Import the EarlyStopping and ModelCheckpoint callbacks
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Early stop on validation accuracy
monitor_val_acc = EarlyStopping(monitor = 'val_acc', patience = 3)

# Save the best model as best_banknote_model.hdf5
modelCheckpoint = ModelCheckpoint('best_banknote_model.hdf5', save_best_only = True)

# Fit your model for a stupid amount of epochs
h_callback = model.fit(X_train, y_train, epochs = 1000000000000, callbacks = [monitor_val_acc, modelCheckpoint], validation_data = (X_test, y_test))

# output:
#     Train on 960 samples, validate on 412 samples
#     Epoch 1/1000000000000
# 
#      32/960 [>.............................] - ETA: 2s - loss: 0.3227 - acc: 0.8438
#     896/960 [===========================>..] - ETA: 0s - loss: 0.2829 - acc: 0.9252
#     960/960 [==============================] - 0s 198us/step - loss: 0.2829 - acc: 0.9250 - val_loss: 0.3030 - val_acc: 0.9126
#     Epoch 2/1000000000000
#     
#      32/960 [>.............................] - ETA: 0s - loss: 0.2523 - acc: 0.9688
#     960/960 [==============================] - 0s 62us/step - loss: 0.2776 - acc: 0.9271 - val_loss: 0.2973 - val_acc: 0.9126
#     Epoch 3/1000000000000
#     
#      32/960 [>.............................] - ETA: 0s - loss: 0.2336 - acc: 0.9688
#     960/960 [==============================] - 0s 61us/step - loss: 0.2726 - acc: 0.9292 - val_loss: 0.2920 - val_acc: 0.9126
#     Epoch 4/1000000000000
#     
#      32/960 [>.............................] - ETA: 0s - loss: 0.2699 - acc: 0.9688
#     960/960 [==============================] - 0s 61us/step - loss: 0.2679 - acc: 0.9312 - val_loss: 0.2870 - val_acc: 0.9126