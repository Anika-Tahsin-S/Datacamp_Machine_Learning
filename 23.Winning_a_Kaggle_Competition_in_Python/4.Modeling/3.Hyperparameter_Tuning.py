## Grid Search
alpha_grid = [0.01, 0.1, 1, 10]

from sklearn.linear_model import Ridge
results = {}

for candidate_alpha in alpha_grid:
    # Create model with specific alpha value
    ridge_regression = Ridge(alpha = candidate_alpha)
    # Find the validation score for model
    # Save the results for each alpha value
    results[candidate_alpha] = validation_score



## ====================================================================================================== ##