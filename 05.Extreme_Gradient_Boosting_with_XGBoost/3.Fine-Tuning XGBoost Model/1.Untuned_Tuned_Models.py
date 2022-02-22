import pandas as pd
import xgboost as xgb
import numpy as np

housing_data = pd.read_csv("ames_housing_trimmed_processed.csv")
X, y = [housing_data[housing_data.columns.tolist()[:-1]],
        housing_data[housing_data.columns.tolist()[:-1]]
housing_dmatrix = xgb.DMatrix(data = X, label = y)



##                   Untuned Model Example                  ##
untuned_params = {"objective":"reg:linear"}
untuned_cv_results_rmse = xgb.cv(dtrain = housing_dmatrix, params = untuned_params,
                            nfold = 4, metrics = "rmse", as_pandas = True, seed = 123)
print("Untuned rmse: %f" %((untuned_cv_results_rmse["test-rmse-mean"]).tail(1)))

##                   Tuned Model Example                  ##
tuned_params = {"objective":"reg:linear", "colsample_bytree": 0.3,
                'learning_rate' : 0.1, 'max_depth': 5}
tuned_cv_results_rmse = xgb.cv(dtrain = housing_dmatrix, params = tuned_params,
                            nfold = 4, num_boost_round = 200, metrics = "rmse",
                            as_pandas = True, seed = 123)
print("Tuned rmse: %f" %((tuned_cv_results_rmse["test-rmse-mean"]).tail(1)))



##                   Tuning The Number of Boosting Rounds                  ##
# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data = X, label = y)

# Create the parameter dictionary for each tree: params 
params = {"objective":"reg:linear", "max_depth":3}

# Create list of number of boosting rounds
num_rounds = [5, 10, 15]

# Empty list to store final round rmse per XGBoost model
final_rmse_per_round = []

# Iterate over num_rounds and build one model per num_boost_round parameter
for curr_num_rounds in num_rounds:

    # Perform cross-validation: cv_results
    cv_results = xgb.cv(dtrain = housing_dmatrix, params = params, 
                        nfold = 3, num_boost_round = curr_num_rounds, 
                        metrics = "rmse", as_pandas = True, seed = 123)
    
    # Append final round RMSE
    final_rmse_per_round.append(cv_results["test-rmse-mean"].tail().values[-1])

# Print the resultant DataFrame
num_rounds_rmses = list(zip(num_rounds, final_rmse_per_round))
print(pd.DataFrame(num_rounds_rmses, columns = ["num_boosting_rounds", "rmse"]))
# As you can see, increasing the number of boosting rounds decreases the RMSE.


##                   Automated Boosting Round Selection Using Early_Stopping                  ##
# Create your housing DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary for each tree: params
params = {"objective":"reg:linear", "max_depth":4}

# Perform cross-validation with early stopping: cv_results
cv_results = xgb.cv(dtrain = housing_dmatrix, params = params, nfold = 3, early_stopping_rounds = 10, num_boost_round = 50, metrics = "rmse", as_pandas = True, seed = 123)

# Print cv_results
print(cv_results)