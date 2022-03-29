# Line curves
training = model.fit(train_data, train_labels, epochs = 3, validation_split = 0.2)

import matplotlob.pyplot as plt
plt.plot(training.history['loss'])
plt.plot(training.history['val_loss'])
plt.show()

# Storing the optimal parameters
from keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint('weights.hdf5', monitor = 'val_loss', save_best_only = True)
callbacks_list = [checkpoint])
model.fit(train_data, train_labels, epochs = 3, validation_split = 0.2, callbacks = callbacks_list)

# Loading stored parameters
model.load_weights('weights.hdf5')
model.predict_classes(test_data)
array([2, 2, 1, 2, 0, 1, 0, 1, 2, 0])







# --------------------------------------------------------------------------------------------------------- #
##                  Plot the learning curves                  ##
import matplotlib.pyplot as plt

# Train the model and store the training object
training = model.fit(train_data, train_labels, epochs = 3, validation_split = 0.2, batch_size = 10)

# Extract the history from the training object
history = training.history

# Plot the training loss 
plt.plot(history['loss'])
# Plot the validation loss
plt.plot(history['val_loss'])

# Show the figure
plt.show()


# output:
#     Epoch 1/3
#     
# 1/4 [======>.......................] - ETA: 3s - loss: 1.1170 - accuracy: 0.40004/4 [==============================] - 2s 194ms/step - loss: 1.0910 - accuracy: 0.5000 - val_loss: 1.0770 - val_accuracy: 0.6000
#     Epoch 2/3
#     
# 1/4 [======>.......................] - ETA: 0s - loss: 1.0896 - accuracy: 0.30004/4 [==============================] - 0s 12ms/step - loss: 1.0560 - accuracy: 0.6500 - val_loss: 1.0535 - val_accuracy: 0.7000
#     Epoch 3/3
#     
# 1/4 [======>.......................] - ETA: 0s - loss: 1.0326 - accuracy: 0.50004/4 [==============================] - 0s 11ms/step - loss: 1.0223 - accuracy: 0.7000 - val_loss: 1.0213 - val_accuracy: 0.9000






##                  Using stored weights to predict in a test set                  ##
# Load the weights from file
model.load_weights('weights.hdf5')

# Predict from the first three images in the test data
model.predict(test_data[0:3])


array([[0.09926841, 0.09344567, 0.8072859 ],
       [0.03939047, 0.043042  , 0.91756755],
       [0.23671192, 0.70986897, 0.05341912],
       [0.08076198, 0.06505188, 0.8541861 ],
       [0.5389348 , 0.42102852, 0.04003668],
       [0.24078315, 0.73240227, 0.02681463],
       [0.732365  , 0.25965095, 0.00798406],
       [0.32749954, 0.44584516, 0.22665524],
       [0.19110951, 0.17169954, 0.6371909 ],
       [0.46978685, 0.36317986, 0.16703328]], dtype = float32)

