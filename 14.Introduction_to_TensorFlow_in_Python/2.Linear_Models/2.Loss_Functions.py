# MSE : Strongly penalizes outliers || High (gradient) sensitivity near minimun
# MAE : Scales linearly with size of error || Low sensitivity near minimum
# Huber : Similar to MSE near minimum || Similar to MAE away from minimun





# --------------------------------------------------------------------------------------------------------- #
##                   Loss functions in TensorFlow                  ##
# Part 1
# Import the keras module from tensorflow
from tensorflow import keras

# Compute the mean squared error (mse)
loss = keras.losses.mse(price, predictions)

# Print the mean squared error (mse)
print(loss.numpy())
# MSE : 141171604777.12717



# Part 2
# Import the keras module from tensorflow
from tensorflow import keras

# Compute the mean absolute error (mae)
loss = keras.losses.mae(price, predictions)

# Print the mean absolute error (mae)
print(loss.numpy())
# MAE : 268827.99302088






##                   Modifying the Loss Function                  ##
# Initialize a variable named scalar
scalar = Variable(1.0, float32)

# Define the model
def model(scalar, features = features):
  	return scalar * features

# Define a loss function
def loss_function(scalar = scalar, features = features, targets = targets):
	# Compute the predicted values
	predictions = model(scalar, features)
    
	# Return the mean absolute error loss
	return keras.losses.mae(targets, predictions)

# Evaluate the loss function and print the loss
print(loss_function(scalar).numpy())
# Output: 3.0