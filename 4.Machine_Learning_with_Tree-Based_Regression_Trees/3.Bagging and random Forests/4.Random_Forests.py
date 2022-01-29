from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.ensemble import RandomForestRegressor

# Set seed for reproducibility
SEED = 2

# Split data into 80% train and 20% test
X_train, y_train, X_test, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = SEED)

# Instantiate a random forest regressor consisting of 400 regression trees.
rf = RandomForestRegressor(n_estimators = 25, min_samples_leaf = 0.12, random_state = SEED)


##                  Evaluate the rf regressor                  ##
# Fit 'rf' to the training set
rf.fit(X_train, y_train)

# Predict test set labels
y_pred = rf.predict(X_test)

# Evaluate and print test-set RMSE
rmse_test = MSE(y_test, y_pred) ** (1/2)
print('Test Set RMSE of rf: {:.2f}'.format(rmse_test))


##                  Visualizing features importances                  ##
import pandas as pd
import matplotlib.pyplot as plt

# Create a pd.Series of features importances
importances_rf = pd.Series(rf.features_importances_, index = X.columns)

# Sort importances_rf and make a horizontal bar plot
sorted_importances_rf = importances_rf.sort.values()
sorted_importances_rf.plot(kind = 'barh', color = 'lightgreen')
plt.title('Features Importances')
plt.show()