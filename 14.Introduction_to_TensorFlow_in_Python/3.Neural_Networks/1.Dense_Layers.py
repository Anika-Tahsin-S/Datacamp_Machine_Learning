# Simple Dense Layer
import tensorflow as tf

inputs = tf.constant([[1, 35]])
weights = tf.Variable([[-0.5], [-0.01]])
bias = tf.Variable([0.5])

# Defining dense layer
product = tf.matmul(inputs, weights)

dense(tf.keras.activations.sigmoid(product + bias))


# Defining a complete model
import tensorflow as tf
inputs = tf.constant(data, tf.float32)

dense1 = tf.keras.layers.Dense(10, activation = 'sigmoid')(inputs)
dense2 = tf.keras.layers.Dense(5, activation = 'sigmoid')(dense1)

output = tf.keras.layers.Dense(1, activation = 'sigmoid')(dense2)



## High Level approach
dense = keras.layers.Dense(10, activation = 'sigmoid')

## Low Level approach
prod = matmul(inputs, weights)
dense = keras.activations.sigmoid(prod)





# --------------------------------------------------------------------------------------------------------- #
##                   The Linear Algebra of Dense Layers                  ##
# Part 1
from tensorflow import Variable, ones, matmul, keras

# Initialize bias1
bias1 = Variable(1.0)

# Initialize weights1 as 3x2 variable of ones
weights1 = Variable(ones((3, 2)))

# Perform matrix multiplication of borrower_features and weights1
product1 = matmul(borrower_features, weights1)

# Apply sigmoid activation function to product1 + bias1
dense1 = keras.activations.sigmoid(product1 + bias1)

# Print shape of dense1
print("\n dense1's output shape: {}".format(dense1.shape))
# output: dense1's output shape: (1, 2)

# Part 2
# From previous step
bias1 = Variable(1.0)
weights1 = Variable(ones((3, 2)))
product1 = matmul(borrower_features, weights1)
dense1 = keras.activations.sigmoid(product1 + bias1)

# Initialize bias2 and weights2
bias2 = Variable(1.0)
weights2 = Variable(ones((2, 1)))

# Perform matrix multiplication of dense1 and weights2
product2 = matmul(dense1, weights2)

# Apply activation to product2 + bias2 and print the prediction
prediction = keras.activations.sigmoid(product2 + bias2)
print('prediction: {}'.format(prediction.numpy()[0,0]))
print('actual: 1')
# output:
# prediction: 0.9525741338729858
# actual: 1







##                   The low-level Approach with Multiple examples                  ##
# Compute the product of borrower_features and weights1
products1 = matmul(borrower_features, weights1)

# Apply a sigmoid activation function to products1 + bias1
dense1 = keras.activations.sigmoid(products1 + bias1)

# Print the shapes of borrower_features, weights1, bias1, and dense1
print('shape of borrower_features: ', borrower_features.shape)
print('shape of weights1: ', weights1.shape)
print('shape of bias1: ', bias1.shape)
print('shape of dense1: ', dense1.shape)
# output:
# shape of borrower_features:  (5, 3)
# shape of weights1:  (3, 2)
# shape of bias1:  (1,)
# shape of dense1:  (5, 2)




##                   Using the Dense Layer Operation                  ##
# Define the first dense layer
dense1 = keras.layers.Dense(7, activation = 'sigmoid')(borrower_features)

# Define a dense layer with 3 output nodes
dense2 = keras.layers.Dense(3, activation = 'sigmoid')(dense1)

# Define a dense layer with 1 output node
predictions = keras.layers.Dense(1, activation = 'sigmoid')(dense2)

# Print the shapes of dense1, dense2, and predictions
print('shape of dense1: ', dense1.shape)
print('shape of dense2: ', dense2.shape)
print('shape of predictions: ', predictions.shape)

# output:
# shape of dense1:  (100, 7)
# shape of dense2:  (100, 3)
# shape of predictions:  (100, 1)