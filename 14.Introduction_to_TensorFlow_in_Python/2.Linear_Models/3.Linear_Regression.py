##                   Set up a Linear Regression                  ##
# Define a linear regression model
def linear_regression(intercept, slope, features = size_log):
	return intercept + features*slope

# Set loss_function() to take the variables as arguments
def loss_function(intercept, slope, features = size_log, targets = price_log):
	# Set the predicted values
	predictions = linear_regression(intercept, slope, features)
    
    # Return the mean squared error loss
	return keras.losses.mse(targets, predictions)

# Compute the loss for different slope and intercept values
print(loss_function(0.1, 0.1).numpy())
print(loss_function(0.1, 0.5).numpy())
# output:
#       145.44653
#       71.866







##                   Train a Linear Model                   ##
# Initialize an Adam optimizer
opt = keras.optimizers.Adam(0.5)

for j in range(100):
	# Apply minimize, pass the loss function, and supply the variables
	opt.minimize(lambda: loss_function(intercept, slope), var_list = [intercept, slope])

	# Print every 10th value of the loss
	if j % 10 == 0:
		print(loss_function(intercept, slope).numpy())

# Plot data and regression line
plot_results(intercept, slope)
# output:
#     9.669482
#     11.726698
#     1.1193314
#     1.6605737
#     0.7982884
#     0.8017316
#     0.6106565
#     0.59997976
#     0.5811015
#     0.5576158







##                   Multiple Linear Regression                   ##
# Define the linear regression model
def linear_regression(params, feature1 = size_log, feature2 = bedrooms):
	return params[0] + feature1*params[1] + feature2*params[2]

# Define the loss function
def loss_function(params, targets = price_log, feature1 = size_log, feature2 = bedrooms):
	# Set the predicted values
	predictions = linear_regression(params, feature1, feature2)
  
	# Use the mean absolute error loss
	return keras.losses.mae(targets, predictions)

# Define the optimize operation
opt = keras.optimizers.Adam()

# Perform minimization and print trainable variables
for j in range(10):
	opt.minimize(lambda: loss_function(params), var_list=[params])
	print_results(params)
# output:
#     loss: 12.418, intercept: 0.101, slope_1: 0.051, slope_2: 0.021
#     loss: 12.404, intercept: 0.102, slope_1: 0.052, slope_2: 0.022
#     loss: 12.391, intercept: 0.103, slope_1: 0.053, slope_2: 0.023
#     loss: 12.377, intercept: 0.104, slope_1: 0.054, slope_2: 0.024
#     loss: 12.364, intercept: 0.105, slope_1: 0.055, slope_2: 0.025
#     loss: 12.351, intercept: 0.106, slope_1: 0.056, slope_2: 0.026
#     loss: 12.337, intercept: 0.107, slope_1: 0.057, slope_2: 0.027
#     loss: 12.324, intercept: 0.108, slope_1: 0.058, slope_2: 0.028
#     loss: 12.311, intercept: 0.109, slope_1: 0.059, slope_2: 0.029
#     loss: 12.297, intercept: 0.110, slope_1: 0.060, slope_2: 0.030