# Build and tune deep learning models using keras
import numpy as np
from keras.layers import Dense
from keras.models import Sequential

predictions = np.loadtxt('predictors_data.csv', delimiter = ',')
n_cols = predictors.shape[1]
model = Sequential()

model.add(Dense(100, activation = 'relu', input_shape = (n_cols,)))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(1))




# --------------------------------------------------------------------------------------------------------- #
##                   Comparing neural network models to classical regression models                  ##
# Which of the models in the diagrams has greater ability to account for interactions?
# Answer: Model 2
# Model 2 has more nodes in the hidden layer, and therefore, greater ability to capture interactions.