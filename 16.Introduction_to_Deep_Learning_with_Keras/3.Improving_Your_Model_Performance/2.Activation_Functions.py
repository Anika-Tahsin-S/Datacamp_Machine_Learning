# Comparing Activations
from keras.models import Sequential
from keras.layers import Dense

# Set a random seed
np.random.seed(1)
# Return a new model with the given activation
def get_model(act_func):
    model = Sequential()
    model.add(Dense(4, input_shape = (2,), activation = act_func))
    model.add(Dense(1, activation = 'sigmoid'))
    return model

# Activation functions to try out
activations = ['relu', 'sigmoid', 'tanh']

# Dictionary to store results
activation_results = {}
for func in activations:
    model = get_model(act_func = func)
    history = model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 1000, verbose = 0)
    activation_results[func] = history

import pandas as pd

val_loss_per_func = {k: v.history['val_loss'] for k, v in activation_results.items()}
val_loss_curves = pd.DataFrame(val_loss_per_func)
val_loss_curves.plot(title = 'Loss Per Activation Function')










# --------------------------------------------------------------------------------------------------------- #
##                   Different activation functions                  ##
# The sigmoid(),tanh(), ReLU(), and leaky_ReLU() functions have been defined and ready for you to use. Each function receives an input number X and returns its corresponding Y value.
# Which of the statements below is false?

# A) The sigmoid() takes a value of 0.5 when X = 0 whilst tanh() takes a value of 0.
# B) The leaky_ReLU() takes a value of -0.01 when X = -1 whilst ReLU() takes a value of 0.
# C) The sigmoid() and tanh() both take values close to -1 for big negative numbers.

# Answer: C. For big negative numbers the sigmoid approaches 0 not -1 whilst the tanh() does take values close to -1.










##                   Comparing activation functions                  ##
# Activation functions to try
activations = ['relu', 'leaky_relu', 'sigmoid', 'tanh']

# Loop over the activation functions
activation_results = {}

for act in activations:
  # Get a new model with the current activation
  model = get_model(act_function = act)
  # Fit the model and store the history results
  h_callback = model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 20, verbose = 0)
  activation_results[act] = h_callback

# output:
#     Finishing with relu ...
#     Finishing with leaky_relu ...
#     Finishing with sigmoid ...
#     Finishing with tanh ...







##                   Comparing activation functions II                  ##
# Create a dataframe from val_loss_per_function
val_loss = pd.DataFrame(val_loss_per_function)

# Call plot on the dataframe
val_loss.plot()
plt.show()

# Create a dataframe from val_acc_per_function
val_acc = pd.DataFrame(val_acc_per_function)

# Call plot on the dataframe
val_acc.plot()
plt.show()

# You've plotted both: loss and accuracy curves. It looks like sigmoid activation worked best for this particular model as the hidden layer's activation function. 
# It led to a model with lower validation loss and higher accuracy after 100 epochs.