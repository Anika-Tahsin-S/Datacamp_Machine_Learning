import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

boston_data = pd.read_csv("boston_housing.csv")

X, y = boston_data.iloc[:,:-1], boston_data.iloc[:,-1] # split the entire dataset into a matrix of samples by features, called X by convention, and a vector of target values, called y by convention.

X_train, y_train, X_test, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)


##                   Decision Trees as Base Learners                  ##
xg_reg =xgb.XGBRegressor(objective = "reg:linear", n_estimators = 10, seed = 123)
xg_reg.fit(X_train, y_train)
preds = xg_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))

print("RMSE: %f" % (rmse))



##                   Linear Base Learners                  ##
# Convert the training and testing sets into DMatrixes: DM_train, DM_test
DM_train = xgb.DMatrix(data = X_train, label = y_train)
DM_test = xgb.DMatrix(data = X_test, label = y_test)

# Create the parameter dictionary: params
params = {"booster" : "gblinear", "objective" : "reg:linear"}

# Train the model: xg_reg
xg_reg = xgb.train(params = params, dtrain = DM_train, num_boost_round = 10)

# Predict the labels of the test set: preds
preds  = xg_reg.predict(DM_test)

rmse = np.sqrt(mean_squarred_error(y_test, preds))

print("RMSE: %f" % (rmse))


##                   Evaluating Model Quality                  ##
# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary: params
params = {"objective":"reg:linear", "max_depth":4}

# Perform cross-validation: cv_results
cv_results = xgb.cv(dtrain = housing_dmatrix, params = params, nfold = 4, 
                    num_boost_round = 5, metrics = "rmse", 
                    as_pandas = True, seed = 123)

# Print cv_results
print(cv_results)

# Extract and print final boosting round metric
print((cv_results["test-rmse-mean"]).tail(1))


## Compute mae
# Perform cross-validation: cv_results
cv_results = xgb.cv(dtrain = housing_dmatrix, params = params, 
                    nfold = 4, num_boost_round = 5, metrics = "mae", 
                    as_pandas = True, seed = 123)

# Print cv_results
print(cv_results)

# Extract and print final boosting round metric
print((cv_results["test-rmse-mean"]).tail(1))