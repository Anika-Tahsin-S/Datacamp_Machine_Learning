# Performing validation set
model.fit(features, labels, epochs = 10, validation_split = 0.20)

## Changing Metric
# Recompile model with accuracy metric
model.compile('adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Train model with validation split
model.fit(features, labels, epochs = 10, validation_split = 0.20)








# --------------------------------------------------------------------------------------------------------- #
##                   Training with Keras                  ##
from tensorflow import keras

# Define a sequential model
model = keras.Sequential()

# Define a hidden layer
model.add(keras.layers.Dense(16, activation = 'relu', input_shape = (784,)))

# Define the output layer
model.add(keras.layers.Dense(4, activation = 'softmax'))

# Compile the model
model.compile('SGD', loss = 'categorical_crossentropy')

# Complete the fitting operation
model.fit(sign_language_features, sign_language_labels, epochs = 5)

# output:
#     Epoch 1/5
#     
#  1/32 [..............................] - ETA: 57s - loss: 1.496827/32 
#       [========================>.....] - ETA: 0s - loss: 1.333932/32 
#       [==============================] - 2s 2ms/step - loss: 1.3230
#     Epoch 2/5
#     
#  1/32 [..............................] - ETA: 0s - loss: 1.449725/32 
#       [======================>.......] - ETA: 0s - loss: 1.190832/32 
#       [==============================] - 0s 2ms/step - loss: 1.1758
#     Epoch 3/5
#     
#  1/32 [..............................] - ETA: 0s - loss: 1.041828/32 
#       [=========================>....] - ETA: 0s - loss: 1.060132/32 
#       [==============================] - 0s 2ms/step - loss: 1.0512
#     Epoch 4/5
#     
#  1/32 [..............................] - ETA: 0s - loss: 0.963727/32 
#       [========================>.....] - ETA: 0s - loss: 0.939432/32 
#       [==============================] - 0s 2ms/step - loss: 0.9392
#     Epoch 5/5
#     
#  1/32 [..............................] - ETA: 0s - loss: 0.876228/32 
#       [=========================>....] - ETA: 0s - loss: 0.850432/32 
#       [==============================] - 0s 2ms/step - loss: 0.8460








##                   Metrics and Validation with Keras                  ##
# Define sequential model
model = keras.Sequential()

# Define the first layer
model.add(keras.layers.Dense(32, activation = 'sigmoid', input_shape = [784,]))

# Add activation function to classifier
model.add(keras.layers.Dense(4, activation = 'softmax'))

# Set the optimizer, loss function, and metrics
model.compile(optimizer = 'RMSprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Add the number of epochs and the validation split
model.fit(sign_language_features, sign_language_labels, epochs = 10, validation_split = 0.1)

# output:
#     Epoch 1/10
#     
#  1/29 [>.............................] - ETA: 1:02 - loss: 1.4231 - accuracy: 0.312526/29 
#       [=========================>....] - ETA: 0s - loss: 1.2747 - accuracy: 0.405029/29 
#       [==============================] - 3s 26ms/step - loss: 1.2651 - accuracy: 0.4171 - val_loss: 1.2242 - val_accuracy: 0.2900
#     Epoch 2/10
#     
#  1/29 [>.............................] - ETA: 0s - loss: 1.3104 - accuracy: 0.250028/29 
#       [===========================>..] - ETA: 0s - loss: 0.9927 - accuracy: 0.684229/29 
#       [==============================] - 0s 3ms/step - loss: 0.9925 - accuracy: 0.6841 - val_loss: 1.0445 - val_accuracy: 0.5900
#     Epoch 3/10
#     
#  1/29 [>.............................] - ETA: 0s - loss: 0.9369 - accuracy: 0.781228/29 
#       [===========================>..] - ETA: 0s - loss: 0.8426 - accuracy: 0.770129/29 
#       [==============================] - 0s 3ms/step - loss: 0.8422 - accuracy: 0.7709 - val_loss: 0.8102 - val_accuracy: 0.7000
#     Epoch 4/10
#     
#  1/29 [>.............................] - ETA: 0s - loss: 0.8435 - accuracy: 0.718826/29 
#       [=========================>....] - ETA: 0s - loss: 0.7021 - accuracy: 0.800529/29 
#       [==============================] - 0s 7ms/step - loss: 0.7009 - accuracy: 0.8098 - val_loss: 0.7993 - val_accuracy: 0.6900
#     Epoch 5/10
#     
#  1/29 [>.............................] - ETA: 0s - loss: 0.7524 - accuracy: 0.781226/29 
#       [=========================>....] - ETA: 0s - loss: 0.6163 - accuracy: 0.860629/29 
#       [==============================] - 0s 3ms/step - loss: 0.6137 - accuracy: 0.8632 - val_loss: 0.6350 - val_accuracy: 0.7300
#     Epoch 6/10
#     
#  1/29 [>.............................] - ETA: 0s - loss: 0.7238 - accuracy: 0.718821/29 
#       [====================>.........] - ETA: 0s - loss: 0.5385 - accuracy: 0.878029/29 
#       [==============================] - 0s 3ms/step - loss: 0.5383 - accuracy: 0.8821 - val_loss: 0.7719 - val_accuracy: 0.5900
#     Epoch 7/10
#     
#  1/29 [>.............................] - ETA: 0s - loss: 0.8442 - accuracy: 0.500025/29 
#       [========================>.....] - ETA: 0s - loss: 0.4801 - accuracy: 0.901329/29 
#       [==============================] - 0s 3ms/step - loss: 0.4764 - accuracy: 0.9077 - val_loss: 0.4521 - val_accuracy: 0.9800
#     Epoch 8/10
#     
#  1/29 [>.............................] - ETA: 0s - loss: 0.4912 - accuracy: 0.937525/29 
#       [========================>.....] - ETA: 0s - loss: 0.4323 - accuracy: 0.923729/29 
#       [==============================] - 0s 3ms/step - loss: 0.4262 - accuracy: 0.9244 - val_loss: 0.5926 - val_accuracy: 0.7100
#     Epoch 9/10
#     
#  1/29 [>.............................] - ETA: 0s - loss: 0.4262 - accuracy: 0.812524/29 
#       [=======================>......] - ETA: 0s - loss: 0.3825 - accuracy: 0.936229/29 
#       [==============================] - 0s 3ms/step - loss: 0.3777 - accuracy: 0.9433 - val_loss: 0.3892 - val_accuracy: 0.9400
#     Epoch 10/10
#     
#  1/29 [>.............................] - ETA: 0s - loss: 0.3319 - accuracy: 1.000023/29 
#       [======================>.......] - ETA: 0s - loss: 0.3509 - accuracy: 0.947029/29 
#       [==============================] - 0s 3ms/step - loss: 0.3376 - accuracy: 0.9511 - val_loss: 0.3641 - val_accuracy: 0.8900











##                   Overfitting Detection                  ##
# Define sequential model
model = keras.Sequential()

# Define the first layer
model.add(keras.layers.Dense(1024, activation = 'relu', input_shape = [784,]))

# Add activation function to classifier
model.add(keras.layers.Dense(4, activation = 'softmax'))

# Finish the model compilation
model.compile(optimizer = keras.optimizers.Adam(lr = 0.001), 
              loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Complete the model fit operation
model.fit(sign_language_features, sign_language_labels, epochs = 50, validation_split = .5)

# output:
#     Epoch 1/50
#    
# 1/1 [==============================] - ETA: 0s - loss: 1.3461 - accuracy: 0.30771/1 [==============================] - 2s 2s/step - loss: 1.3461 - accuracy: 0.3077 - val_loss: 3.0842 - val_accuracy: 0.3846
# 1/1 [==============================] - 3s 3s/step - loss: 1.3461 - accuracy: 0.3077 - val_loss: 3.0842 - val_accuracy: 0.3846
#     Epoch 2/50
#     
# 1/1 [==============================] - ETA: 0s - loss: 2.4663 - accuracy: 0.23081/1 [==============================] - 0s 38ms/step - loss: 2.4663 - accuracy: 0.2308 - val_loss: 4.2784 - val_accuracy: 0.3846
#     Epoch 3/50
#     
# 1/1 [==============================] - ETA: 0s - loss: 2.1878 - accuracy: 0.61541/1 [==============================] - 0s 47ms/step - loss: 2.1878 - accuracy: 0.6154 - val_loss: 5.6272 - val_accuracy: 0.3077
#     Epoch 4/50
#     
# 1/1 [==============================] - ETA: 0s - loss: 3.6705 - accuracy: 0.38461/1 [==============================] - 0s 24ms/step - loss: 3.6705 - accuracy: 0.3846 - val_loss: 4.5968 - val_accuracy: 0.3077
#     Epoch 5/50
#     
# 1/1 [==============================] - ETA: 0s - loss: 2.5616 - accuracy: 0.69231/1 [==============================] - 0s 32ms/step - loss: 2.5616 - accuracy: 0.6923 - val_loss: 4.3052 - val_accuracy: 0.0769
#     Epoch 6/50
#     
# 1/1 [==============================] - ETA: 0s - loss: 2.3775 - accuracy: 0.61541/1 [==============================] - 0s 32ms/step - loss: 2.3775 - accuracy: 0.6154 - val_loss: 2.8193 - val_accuracy: 0.0769
#     Epoch 7/50
#     
# 1/1 [==============================] - ETA: 0s - loss: 1.5247 - accuracy: 0.61541/1 [==============================] - 0s 38ms/step - loss: 1.5247 - accuracy: 0.6154 - val_loss: 1.0500 - val_accuracy: 0.5385
# ..........................................................................

















##                   Evaluating Models                  ##
# Two models have been trained and are available: large_model, which has many parameters; and small_model, which has fewer parameters. Both models have been trained using train_features and train_labels, which are available to you. A separate test set, which consists of test_features and test_labels, is also available.

# Evaluate the small model using the train data
small_train = small_model.evaluate(train_features, train_labels)

# Evaluate the small model using the test data
small_test = small_model.evaluate(test_features, test_labels)

# Evaluate the large model using the train data
large_train = large_model.evaluate(train_features, train_labels)

# Evaluate the large model using the test data
large_test = large_model.evaluate(test_features, test_labels)

# Print losses
print('\n Small - Train: {}, Test: {}'.format(small_train, small_test))
print('Large - Train: {}, Test: {}'.format(large_train, large_test))

# output:
    
# 1/4 [======>.......................] - ETA: 1s - loss: 0.17384/4 [==============================] - 1s 2ms/step - loss: 0.1698
#     
# 1/4 [======>.......................] - ETA: 0s - loss: 0.32514/4 [==============================] - 0s 3ms/step - loss: 0.2849
#     
# 1/4 [======>.......................] - ETA: 1s - loss: 0.04254/4 [==============================] - 0s 2ms/step - loss: 0.0396
#     
# 1/4 [======>.......................] - ETA: 0s - loss: 0.14144/4 [==============================] - 0s 2ms/step - loss: 0.1454
#     
#      Small - Train: 0.16981548070907593, Test: 0.28487256169319153
#     Large - Train: 0.03957207500934601, Test: 0.14543524384498596