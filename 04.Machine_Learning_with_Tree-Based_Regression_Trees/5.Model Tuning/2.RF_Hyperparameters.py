from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as MSE

# Set seed for reproducibility
SEED = 1

# Split data into 80% train and 20% test
X_train, y_train, X_test, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = SEED)

# Instantiate a random forests regressor 'rf'
rf = DecisionTreeCLassifier(random_state = SEED)



##                   Set The Tree's Hyperparameter Grid                  ##
# Define a grid of hyperparameter 'params_rf'
params_rf = {
                'n_estimators': [100, 350, 500],
                'max_depth': [4, 6, 8],
                'min_samples_leaf': [2, 10, 30],
                'max_features': ['log2', 'auto', 'sqrt']
            }

##                   Search For The Optimal Tree                  ##
# Instantiate 'grid_rf'
# The parameter verbose controls verbosity; the higher its value, the more messages are printed during fitting. 
grid_rf = GridSearchCV(estimator = rf,
                    param_grid = params_rf, 
                    cv = 3, 
                    scoring = 'neg_mean_squared_error', 
                    verbose = 1, 
                    n_jobs = -1)

# Fit 'rf' to the training set
grid_rf.fit(X_train, y_train)

# Extract best hyperparameters from 'rf'
best_hyperparams = grid_rf.best_params_
print('Best hyperparameteres:\n', best_hyperparams)



##                   Evaluate The Optimal Tree                  ##
# Extract best model from 'grid_rf'
best_model = grid_rf.best_estimator_

# Predict the test set labels
y_pred = best_model.predict(X_test)

# Evaluate the test set RMSE
rmse_test = MSE(y_test, y_pred) ** (1/2)

# Print test set RMSE
print('Test set RMSE of score: {:.2f}'.format(rmse_test))