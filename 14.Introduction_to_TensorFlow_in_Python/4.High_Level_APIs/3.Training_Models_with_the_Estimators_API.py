##                   Preparing to Train with Estimators                  ##
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column


# Define feature columns for bedrooms and bathrooms
bedrooms = feature_column.numeric_column("bedrooms")
bathrooms = feature_column.numeric_column("bathrooms")

# Define the list of feature columns
feature_list = [bedrooms, bathrooms]

def input_fn():
	# Define the labels
	labels = np.array(housing.price)
	# Define the features
	features = {'bedrooms':np.array(housing['bedrooms']), 
                'bathrooms':np.array(housing['bathrooms'])}
	return features, labels








##                   Defining Estimators                  ##
# Part 1
# Define the model and set the number of steps
model = estimator.DNNRegressor(feature_columns = feature_list, hidden_units = [2, 2])
model.train(input_fn, steps = 1)

# Part 2
# Define the model and set the number of steps
model = estimator.LinearRegressor(feature_columns = feature_list)
model.train(input_fn, steps = 2)