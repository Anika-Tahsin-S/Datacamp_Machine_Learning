# Model Specification
import numpy as np
from keras.layers import Dense
from keras.models import Sequential

pred = np.loadtxt('predictions_data.csv', delimiter = ',')
n_cols = pred.shape[1]

model = Sequential()
model.add(Dense(100, activation = 'relu', input_shape = (n_cols, ))
model.add(Dense(100, activation = 'relu')
model.add(Dense(1))





# --------------------------------------------------------------------------------------------------------- #
##                   Understanding your Data                  ##
# The data is pre-loaded into a pandas DataFrame called df.
# The target variable you'll be predicting is wage_per_hour. Some of the predictor variables are binary indicators, where a value of 1 represents True, and 0 represents False.

# Of the 9 predictor variables in the DataFrame, how many are binary indicators? The min and max values as shown by .describe() will be informative here. How many binary indicator predictors are there?
# Answer: 6. There are 6 binary indicators.





##                   Specifying a Model                  ##
# Import necessary modules
import keras
from keras.layers import Dense
from keras.models import Sequential

# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]

# Set up the model: model
model = Sequential()

# Add the first layer
model.add(Dense(50, activation = 'relu', input_shape = (n_cols, )))

# Add the second layer
model.add(Dense(32, activation = 'relu'))

# Add the output layer
model.add(Dense(1))