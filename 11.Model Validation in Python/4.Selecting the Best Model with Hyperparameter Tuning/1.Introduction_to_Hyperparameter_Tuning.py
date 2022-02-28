##                   Creating Hyperparameters                  ##
# Part 1
# Review the parameters of rfr
print(rfr.get_params())
# output:
# {'bootstrap': True, 'ccp_alpha': 0.0, 'criterion': 'squared_error', 'max_depth': None, 'max_features': 'auto', 'max_leaf_nodes': None, 'max_samples': None, 
# 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 100, 'n_jobs': None, 
# 'oob_score': False, 'random_state': 1111, 'verbose': 0, 'warm_start': False}


# Part 2
# Maximum Depth
max_depth = [4, 8, 12]

# Minimum samples for a split
min_samples_split = [2, 5, 10]


# Part 3
# Max features 
max_features = [4, 6, 8, 10]





##                   Running a Model Using Ranges                  ##
from sklearn.ensemble import RandomForestRegressor

# Fill in rfr using your variables
rfr = RandomForestRegressor(
    n_estimators = 100,
    max_depth = random.choice(max_depth),
    min_samples_split = random.choice(min_samples_split),
    max_features = random.choice(max_features))

# Print out the parameters
print(rfr.get_params())

# output: {'bootstrap': True, 'ccp_alpha': 0.0, 'criterion': 'squared_error', 'max_depth': 4, 'max_features': 10, 'max_leaf_nodes': None, 'max_samples': None, 
# 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 100, 'n_jobs': None, 
# 'oob_score': False, 'random_state': None, 'verbose': 0, 'warm_start': False}