import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

boston_data = pd.read_csv("boston_housing.csv")

X, y = boston_data.iloc[:,:-1], boston_data.iloc[:,-1] # split the entire dataset into a matrix of samples by features, called X by convention, and a vector of target values, called y by convention.



##                   Using Regularization in XGBoost                  ##
housing_dmatrix = xgb.DMatrix(data = X, label = y)

# Create the parameter dictionary: params
params = {"objective" : "reg:linear", "max_depth" : 3}


reg_params = [1, 10, 100] # We create a list of 3 different l1 or alpha values that we will try
rmse_l2 = []

for reg in reg_params:
    params['lambda'] = reg
    cv_results = xgb.cv(dtrain = housing_dmatrix, params = params, nfold = 2,
                        num_boost_run = 5, metrics = "rmse",
                        as_pandas = True, seed = 123)
    rmse_l2.append(cv_results["test-rmse-mean"].tail(1).value[0])

print("Best rmse as a function of l2:")
print(pd.DataFrame(list(zip(reg_params, rmse_l2)), columns = ["l2", "rmse"]))

# It looks like as as the value of 'lambda' increases, so does the RMSE.



##                   Visualizing Individual XGBoost Trees                  ##
# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary: params
params = {"objective":"reg:linear", "max_depth":2}

# Train the model: xg_reg
xg_reg = xgb.train(params = params, dtrain = housing_dmatrix, num_boost_round = 10)

# Plot the first tree
xgb.plot_tree(xg_reg, num_trees = 0)
plt.show()

# Plot the fifth tree
xgb.plot_tree(xg_reg, num_trees = 4)
plt.show()

# Plot the last tree sideways
xgb.plot_tree(xg_reg, num_trees = 9, rankdir = "LR")
plt.show()



##                   Visualizing Feature Importances: What Features Are Most Important In My Dataset                  ##
# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data = X, label = y)

# Create the parameter dictionary: params
params = {"objective":"reg:linear", "max_depth":4}

# Train the model: xg_reg
xg_reg = xgb.train(params = params, dtrain = housing_dmatrix, num_boost_round = 10)

# Plot the feature importances
xgb.plot_importance(xg_reg)
plt.show()