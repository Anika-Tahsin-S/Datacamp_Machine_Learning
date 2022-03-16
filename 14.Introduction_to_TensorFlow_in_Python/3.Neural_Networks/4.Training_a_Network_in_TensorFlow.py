# Initializing variables in TensorFlow
import tensorflow as tf

# random normal variable
weights = tf.Variable(tf.random.normal([500, 500]))

# truncated random normal variable
weights = tf.Variable(tf.random.truncated_normal([500, 500]))

dense = tf.keras.layers.Dense(32, activation = 'relu')
dense = tf.keras.layers.Dense(32, activation = 'relu', kernel_initializer = 'zeros')

## Implementing dropouts
import numpy as np
import tensorflow as tf

inputs = np.array(borrower_features, np.float32)
dense1 = tf.keras.layers.Dense(32, activation = 'relu')(inputs)
dense2 = tf.keras.layers.Dense(16, activation = 'relu')(dense1)
dropout1 = tf.keras.layers.Dropout(0.25)(dense2)
outputs = tf.keras,layers.Dense(1, activation = 'sigmoid')(dropout1)







# --------------------------------------------------------------------------------------------------------- #
##                   Initialization in TensorFlow                  ##
from tensorflow import Variable, ones

# Define the layer 1 weights
w1 = Variable(random.normal([23, 7]))

# Initialize the layer 1 bias
b1 = Variable(ones([7]))

# Define the layer 2 weights
w2 = Variable(random.normal([7, 1]))

# Define the layer 2 bias
b2 = Variable(0.0)






##                   Defining the Model and Loss Function                  ##
# Define the model
def model(w1, b1, w2, b2, features = borrower_features):
	# Apply relu activation functions to layer 1
	layer1 = keras.activations.relu(matmul(features, w1) + b1)
    # Apply dropout rate of 0.25
	dropout = keras.layers.Dropout(0.25)(layer1)
	return keras.activations.sigmoid(matmul(dropout, w2) + b2)

# Define the loss function
def loss_function(w1, b1, w2, b2, features = borrower_features, targets = default):
	predictions = model(w1, b1, w2, b2)
	# Pass targets and predictions to the cross entropy loss
	return keras.losses.binary_crossentropy(targets, predictions)






##                   Training Neural Networks with TensorFlow                  ##
# Train the model
for j in range(100):
    # Complete the optimizer
	opt.minimize(lambda: loss_function(w1, b1, w2, b2), 
                 var_list = [w1, b1, w2, b2])

# Make predictions with model using test features
model_predictions = model(w1, b1, w2, b2, test_features)

# Construct the confusion matrix
confusion_matrix(test_targets, model_predictions)