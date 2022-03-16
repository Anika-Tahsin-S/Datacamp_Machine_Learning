## Gradient descent optimizer
# tf.keras.optimizers.SGD()
# learning_rate : typically between 0.5 - 0.001; which will determine how quicly the model parameter adjust during training

# A complete example
import tensorflow as tf

# Define model function
def model(bias, weights, features = borrower_features):
    product = tf.matmul(features, weights)
    return tf.keras.activations.sigmoid(product + bias)

# Compute the predicted values and loss
def loss_function(bias, weights, targets - default, features = borrower_features):
    predictions = model(bias, weights)
    return tf.keras.losses.binary_crossentropy(targets, predictions)

# Minimize the loss function with RMS propagation
opt = tf.keras.optimizers.RMSprop(learning_rate = 0.01, momentum = 0.9)
opt.minimize(lambda: loss_function(bias, weights), var_list = [bias, weights])







# --------------------------------------------------------------------------------------------------------- #
##                   The Dangers of Local Minima                  ##
import numpy
# Initialize x_1 and x_2
x_1 = Variable(6.0, float32)
x_2 = Variable(0.3, float32)

# Define the optimization operation
opt = keras.optimizers.SGD(learning_rate = 0.01)

for j in range(100):
	# Perform minimization using the loss function and x_1
	opt.minimize(lambda: loss_function(x_1), var_list = [x_1])
	# Perform minimization using the loss function and x_2
	opt.minimize(lambda: loss_function(x_2), var_list = [x_2])

# Print x_1 and x_2 as numpy arrays
print(x_1.numpy(), x_2.numpy())
# output: 4.3801394 0.42052683





##                   Avoiding Local Minima                  ##
# Initialize x_1 and x_2
x_1 = Variable(0.05, float32)
x_2 = Variable(0.05, float32)

# Define the optimization operation for opt_1 and opt_2
opt_1 = keras.optimizers.RMSprop(learning_rate = 0.01, momentum = 0.99)
opt_2 = keras.optimizers.RMSprop(learning_rate = 0.01, momentum = 0.00)

for j in range(100):
	opt_1.minimize(lambda: loss_function(x_1), var_list = [x_1])
    # Define the minimization operation for opt_2
	opt_2.minimize(lambda: loss_function(x_2), var_list = [x_2])

# Print x_1 and x_2 as numpy arrays
print(x_1.numpy(), x_2.numpy())
# output: 4.3150263 0.4205261