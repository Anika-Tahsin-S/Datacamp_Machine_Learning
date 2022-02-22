from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.ensemble import GradientBoostingRegressor

# Set seed for reproducibility
SEED = 2

# Split data into 80% train and 20% test
X_train, y_train, X_test, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = SEED)


##                   Define The GB Regressor                  ##
# Instantiate a GradientBoostingRegressor 'gbt'
gbt = GradientBoostingRegressor(n_estimators = 200, max_depth = 4, random_state = SEED)



##                   Train The GB regressor                  ##
# Fit 'adb_clf' to the training set
gbt.fit(X_train, y_train)

# Predict the test set labels
y_pred = gbt.predict(X_test)



##                   Evaluate The SGB regressor                  ##
# Compute MSE
mse_test = MSE(y_test, y_pred)

# Evaluate and print test-set RMSE
rmse_test = mse_test ** (1/2)
print('Test set RMSE: {:.3f}'.format(rmse_test))