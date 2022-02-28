##                   scikit-learn's methods                  ##
# Instruction 1: Load the cross-validation method
from sklearn.model_selection import cross_val_score

# Instruction 2: Load the random forest regression model
from sklearn.ensemble import RandomForestRegressor

# Instruction 3: Load the mean squared error method
# Instruction 4: Load the function for creating a scorer
from sklearn.metrics import mean_squared_error, make_scorer




##                   Implement cross_val_score()                  ##
rfc = RandomForestRegressor(n_estimators = 25, random_state = 1111)
mse = make_scorer(mean_squared_error)

# Set up cross_val_score
cv = cross_val_score(estimator = rfc,
                     X = X_train,
                     y = y_train,
                     cv = 10,
                     scoring = mse)

# Print the mean error
print(cv.mean())
# output: 155.4061992697056