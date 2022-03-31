# Key differences from grid search
# n_iter: which is the number of samples for the random search to take from your grid. 
# param_distributions: is slightly different from param_grid. We can optionally give information on how to sample such as using a particular distribution we provide. 
# If we just give a list as we have been doing, the default is to sample 'uniformly' meaning every item in the list (combination) has equal chance of being chosen.  



from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV

learn_rate_list = np.linspace(0.001, 2, 150)
min_sample_leaf_list = list(range(1, 51))

# Create the parameter grid
param_grid = {'learning_rate': learn_rate_list, 
              'min_samples_leaf': min_sample_leaf_list}

number_model = 10

# Create a random search object
random_GBM_class = RandomizedSearchCV(
    estimator = GradientBoostingClassifier(),
    param_distributions = param_grid,
    n_iter = number_model,
    scoring = 'accuracy', n_jobs = 4, cv = 10, refit = True, return_train_score = True
)

# Fit to the training data
random_GBM_class.fit(X_train, y_train)

# Print the values used for both hyperparameters
print(random_GBM_class.cv_results_['param_learning_rate'])
print(random_GBM_class.cv_results_['param_min_samples_leaf'])

# Plot
X_lims = [np.min(learn_rate_list), np.max(learn_rate_list)]
y_lims = [np.min(min_sample_leaf_list), np.max(min_sample_leaf_list)]

plt.scatter(randy, rand_x, c = ['blue'] * 10)
plt.gca().set(xlabel = 'learn_rate', ylabel = 'min_samples_leaf', title = 'Random Search Hyperparameters')
plt.show()







# --------------------------------------------------------------------------------------------------------- #
##                  RandomSearchCV inputs                  ##
# Confirm how many hyperparameter combinations & print
number_combs = len(combinations_list)
print(number_combs)

# Sample and visualise specified combinations
for x in [50, 500, 1500]:
    sample_and_visualize_hyperparameters(x)
    
# Sample all the hyperparameter combinations & visualise
sample_and_visualize_hyperparameters(x)

# Which of these parameters is only for a RandomizedSearchCV?
# Answer: Which of these parameters is only for a RandomizedSearchCV?
# RandomizedSearchCV asks you for how many models to sample from the grid you set.




##                  The RandomizedSearchCV Object                  ##
# Create the parameter grid
param_grid = {'learning_rate': np.linspace(0.1, 2, 150), 'min_samples_leaf': list(range(20,65))} 

# Create a random search object
random_GBM_class = RandomizedSearchCV(
    estimator=GradientBoostingClassifier(),
    param_distributions=param_grid,
    n_iter=10,
    scoring='accuracy', n_jobs=4, cv=5, refit=True, return_train_score=True
)

# Fit to the training data
random_GBM_class.fit(X_train, y_train)

# Print the values used for both hyperparameters
print(random_GBM_class.cv_results_['param_learning_rate'])
print(random_GBM_class.cv_results_['param_min_samples_leaf'])

# output:
#     [1.1073825503355705 1.0691275167785235 0.4697986577181208
#      1.2476510067114095 1.5664429530201343 1.7577181208053692
#      1.859731543624161 1.5791946308724834 0.5463087248322147
#      1.7577181208053692]
#     [47 54 61 30 63 32 60 43 38 27]






##                  RandomSearchCV in Scikit Learn                  ##
# Create the parameter grid
param_grid = {'max_depth': list(range(5,26)), 'max_features': ['auto', 'sqrt']} 

# Create a random search object
random_rf_class = RandomizedSearchCV(
    estimator = RandomForestClassifier(n_estimators = 80),
    param_distributions = param_grid, n_iter = 5,
    scoring = 'roc_auc', n_jobs = 4, cv = 3, refit = True, return_train_score = True )

# Fit to the training data
random_rf_class.fit(X_train, y_train)

# Print the values used for both hyperparameters
print(random_rf_class.cv_results_['param_max_depth'])
print(random_rf_class.cv_results_['param_max_features'])

# output:
#     [18 11 10 22 10]
#     ['sqrt' 'auto' 'sqrt' 'sqrt' 'auto']

