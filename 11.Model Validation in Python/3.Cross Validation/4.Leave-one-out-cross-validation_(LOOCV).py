##                   When to use LOOCV                  ##
rfc = RandomForestRegressor(n_estimators = 25, random_state = 1111)
mse = make_scorer(mean_squared_error)

# Set up cross_val_score
cv = cross_val_score(estimator = rfc,
                     X = X_train,
                     y = y_train,
                     cv = 10,
                     scoring = mse)

# Print the mean error
print(cv.mean_squared_error())

print(X.shape)
# (122624, 27)

# Which of the following are reasons you might NOT run LOOCV on the provided X dataset? The X data has been loaded for you to explore as you see fit.
#     A: The X dataset has 122,624 data points, which might be computationally expensive and slow.
#     B: You cannot run LOOCV on classification problems.
#     C: You want to test different values for 15 different parameters
# Answer: A & C
# This many observations will definitely slow things down and could be computationally expensive. 
# If you don't have time to wait while your computer runs through 1,000 models, you might want to use 5 or 10-fold cross-validation.




##                   Leave-one-out-cross-validation                  ##
import numpy as np
from sklearn.metrics import mean_absolute_error, make_scorer

# Create scorer
mae_scorer = make_scorer(mean_absolute_error)

rfr = RandomForestRegressor(n_estimators = 15, random_state = 1111)

# Implement LOOCV
scores = cross_val_score(rfr, X = X, y = y, cv = X.shape[0], scoring = mae_scorer)

# Print the mean and standard deviation
print("The mean of the errors is: %s." % np.mean(scores))
print("The standard deviation of the errors is: %s." % np.std(scores))
# output:
#     The mean of the errors is: 9.52044832324183.
#     The standard deviation of the errors is: 7.349020637882744.