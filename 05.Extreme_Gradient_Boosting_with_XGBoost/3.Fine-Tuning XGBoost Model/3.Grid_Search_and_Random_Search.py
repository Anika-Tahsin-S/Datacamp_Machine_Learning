##                   Grid Search                  ##
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import GridSearchCv

housing_data = pd.read_csv("ames_housing_trimmed_processed.csv")
X, y = [housing_data[housing_data.columns.tolist()[:-1]],
        housing_data[housing_data.columns.tolist()[:-1]]

housing_dmatrix = xgb.DMatrix(data = X, label = y)
gbm_param_grid = {'learning_rate': [0.01, 0.1, 0.5, 0.9],
                    'n_estimator' : [0.3, 0.5, 0.9]}

gbm = xgb.XGBRegressor()
gris_mse = GridSearchCv(estimator = gbm, param_grid = gbm_param_grid, 
                        scoring = 'neg_mean_squared_error', cv = 4, verbose = 1)

grid_mse.fit(X, y)
print("Best parametes found: ", grid_mse.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(grid_mse.best_score_)))



##                   Random Search                  ##
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import RandomizedSearchCv

housing_data = pd.read_csv("ames_housing_trimmed_processed.csv")
X, y = [housing_data[housing_data.columns.tolist()[:-1]],
        housing_data[housing_data.columns.tolist()[:-1]]

housing_dmatrix = xgb.DMatrix(data = X, label = y)
gbm_param_grid = {'learning_rate': np.arange(0.05, 1.05, 0.05),
                    'n_estimator' : [200],
                    'subsample' : np.arange(0.05, 1.05, 0.05)}

gbm = xgb.XGBRegressor()
randomized_mse = RandomizedSearchCv(estimator = gbm, param_distributions = gbm_param_grid, 
                    n_iter = 25, scoring = 'neg_mean_squared_error', cv = 4, verbose = 1)

randomized_mse.fit(X, y)
print("Best parametes found: ", randomized_mse.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(randomized_mse.best_score_)))



##                   Grid Search with XGBoost                  ##
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import GridSearchCv

housing_data = pd.read_csv("ames_housing_trimmed_processed.csv")
X, y = [housing_data[housing_data.columns.tolist()[:-1]],
        housing_data[housing_data.columns.tolist()[:-1]]

housing_dmatrix = xgb.DMatrix(data = X, label = y)
# Create the parameter grid: gbm_param_grid
gbm_param_grid = {
    'colsample_bytree': [0.3, 0.7],
    'n_estimators': [50],
    'max_depth': [2, 5]
}

# Instantiate the regressor: gbm
gbm = xgb.XGBRegressor()

# Perform grid search: grid_mse
grid_mse = GridSearchCV(estimator = gbm, param_grid = gbm_param_grid, scoring = "neg_mean_squared_error", cv = 4, verbose = 1)


# Fit grid_mse to the data
grid_mse.fit(X, y)

# Print the best parameters and lowest RMSE
print("Best parameters found: ", grid_mse.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(grid_mse.best_score_)))




##                   Random Search with XGBoost                  ##
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import RandomizedSearchCv

housing_data = pd.read_csv("ames_housing_trimmed_processed.csv")
X, y = [housing_data[housing_data.columns.tolist()[:-1]],
        housing_data[housing_data.columns.tolist()[:-1]]

housing_dmatrix = xgb.DMatrix(data = X, label = y)
# Create the parameter grid: gbm_param_grid 
gbm_param_grid = {
    'n_estimators': [25],
    'max_depth': range(2, 12)
}

# Instantiate the regressor: gbm
gbm = xgb.XGBRegressor(n_estimators = 10)

# Perform random search: grid_mse
randomized_mse = RandomizedSearchCV(estimator = gbm, param_distributions = gbm_param_grid, n_iter = 5, scoring = 'neg_mean_squared_error', cv = 4, verbose = 1)


# Fit randomized_mse to the data
randomized_mse.fit(X, y)

# Print the best parameters and lowest RMSE
print("Best parameters found: ", randomized_mse.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(randomized_mse.best_score_)))