from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.ensemble import GradientBoostingRegressor

# Set seed for reproducibility
SEED = 2

# Split data into 80% train and 20% test
X_train, y_train, X_test, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = SEED)

##                   Regression with SGB                  ##
# Instantiate a GradientBoostingRegressor 'sgbt'
# the parameter subsample was set to 0.9 in order for each tree to sample 80% of the data for training.
# The parameter max_features was set to 0.75 so that each tree uses 20% of available features to perform the best-split.
sgbr = GradientBoostingRegressor(max_depth = 4, subsample = 0.9, max_features = 0.75, n_estimators = 200,  random_state = SEED)



##                   Train The SGB regressor                  ##
# Fit 'adb_clf' to the training set
sgbr.fit(X_train, y_train)

# Predict the test set labels
y_pred = sgbr.predict(X_test)



##                   Evaluate The SGB regressor                  ##
# Compute MSE
mse_test = MSE(y_test, y_pred)

# Evaluate and print test-set RMSE
rmse_test = mse_test ** (1/2)
print('Test set RMSE: {:.3f}'.format(rmse_test))
# Test set RMSE of sgbr: 49.979
# The stochastic gradient boosting regressor achieves a lower test set RMSE than the gradient boosting regressor (which was 52.065)!