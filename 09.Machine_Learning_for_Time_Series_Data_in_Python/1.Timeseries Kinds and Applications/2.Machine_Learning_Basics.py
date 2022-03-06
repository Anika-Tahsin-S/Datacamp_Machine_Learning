##                   Machine Learning Basics                  ##
# Always take a look at raw data #
from turtle import shape
import numpy as np
array.shape
arra[:3]

import pandas as pd
df.head()

# Always visualize the data #
import matplotlib as plt
fig, ax = plt.sybplots()
ax.plot(....)

# Using Pandas
fig, ax = plt.subplots()
df.plot(...., ax = ax)

# If not shaped properly, use Tranpose
array.T.shape

# Another option: reshape
array.reshape([-1, 1],shape)

# Modeling. Support Vector Machines to classify datapoints 
from sklearn.svm import LinearSVC

model = LinearSVC()
model.fit(X, y)

# Invastiagte the patterns the model has found
model.coef_

predictions = model.predict(X_test)

# ---------------------------------------------------------




##                   Fitting a Simple Model: Classification                  ##
# Print the first 5 rows for inspection
print(data.head())

# Part 2
from sklearn.svm import LinearSVC

# Construct data for the model
X = data[["petal length (cm)", "petal width (cm)"]]
y = data[['target']]

# Fit the model
model = LinearSVC()
model.fit(X, y)


##                   Predicting Using a Classification Model                  ##
# Create input array
X_predict = targets[['petal length (cm)', 'petal width (cm)']]

# Predict with the model
predictions = model.predict(X_predict)
print(predictions)

# Visualize predictions and actual values
plt.scatter(X_predict['petal length (cm)'], X_predict['petal width (cm)'],
            c = predictions, cmap = plt.cm.coolwarm)
plt.title("Predicted class values")
plt.show()


##                   Fitting a Simple Model: Regression                  ##
from sklearn import linear_model

# Prepare input and output DataFrames
X = boston[["AGE"]]
y = boston[["RM"]]

# Fit the model
model = linear_model.LinearRegression()
model.fit(X, y)


##                   Predicting Using a Regression Model                  ##
# Generate predictions with the model using those inputs
predictions = model.predict(new_inputs.reshape(-1, 1))

# Visualize the inputs and predicted values
plt.scatter(new_inputs, predictions, color='r', s=3)
plt.xlabel('inputs')
plt.ylabel('predictions')
plt.show()