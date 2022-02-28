# estimator: the model to use
# param_distributions: dictionary containing hyperparameters and possible values
# n_iter: number of iterations
# scoring: scoring method to use




# ------------------------------------------------------------------- #
##                   Preparing for RandomizedSearch                  ##
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error

# Finish the dictionary by adding the max_depth parameter
param_dist = {"max_depth": [2, 4, 6, 8],
              "max_features": [2, 4, 6, 8, 10],
              "min_samples_split": [2, 4, 8, 16]}

# Create a random forest regression model
rfr = RandomForestRegressor(n_estimators = 10, random_state = 1111)

# Create a scorer to use (use the mean squared error)
scorer = make_scorer(mean_squared_error)




##                   Implementing RandomizedSearchCV                  ##
# Import the method for random search
from sklearn.model_selection import RandomizedSearchCV

# Build a random search using param_dist, rfr, and scorer
random_search =RandomizedSearchCV(
                    estimator = rfr,
                    param_distributions = param_dist,
                    n_iter = 10,
                    cv = 5,
                    scoring = scorer)