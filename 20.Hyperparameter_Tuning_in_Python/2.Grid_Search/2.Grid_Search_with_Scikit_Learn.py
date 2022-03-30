# Building GridSearchCV Object
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Create the grid
param_grid = {
    'max_depth': [2, 4, 6, 8],
    'min_samples_leaf': [1, 2, 4, 6]
}


# Base Classifier with some set parameters
rf_class = RandomForestClassifier(criterion = 'entropy', max_features = 'auto')

# Create a GridSearchCV object
grid_rf_class = GridSearchCV(estimator = rf_class, param_grid = param_grid,
                            scoring = 'roc_auc', n_jobs = 4, cv = 5, 
                            refit = True, return_train_score = True)

grid_rf_class.fit(X_train, y_train)
grid_rf_class.predict(X_test)









# --------------------------------------------------------------------------------------------------------- #
##                  GridSearchCV inputs                  ##
Model #1:
 GridSearchCV(
    estimator = RandomForestClassifier(),
    param_grid = {'max_depth': [2, 4, 8, 15], 'max_features': ['auto', 'sqrt']},
    scoring='roc_auc',
    n_jobs=4,
    cv=5,
    refit=True, return_train_score=True) 


Model #2:
 GridSearchCV(
    estimator = KNeighborsClassifier(),
    param_grid = {'n_neighbors': [5, 10, 20], 'algorithm': ['ball_tree', 'brute']},
    scoring='accuracy',
    n_jobs=8,
    cv=10,
    refit=False) 


Model #3:
 GridSearchCV(
    estimator = GradientBoostingClassifier(),
    param_grid = {'number_attempts': [2, 4, 6], 'max_depth': [3, 6, 9, 12]},
    scoring='accuracy',
    n_jobs=2,
    cv=7,
    refit=True) 

# Which of these GridSearchCV objects would not work when we try to fit it?
# Answer: model_3 would not work when we try to fit it.
# Which of these GridSearchCV objects would not work when we try to fit it?





##                  GridSearchCV with Scikit Learn                  ##
# Create a Random Forest Classifier with specified criterion
rf_class = RandomForestClassifier(criterion = 'entropy')

# Create the parameter grid
param_grid = {'max_depth': [2, 4, 8, 15], 'max_features': ['auto', 'sqrt']} 

# Create a GridSearchCV object
grid_rf_class = GridSearchCV(
    estimator = rf_class,
    param_grid = param_grid,
    scoring = 'roc_auc',
    n_jobs = 4,
    cv = 5,
    refit = True, return_train_score = True)
print(grid_rf_class)

# output:
    GridSearchCV(cv=5, error_score='raise-deprecating',
           estimator=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators='warn', n_jobs=None,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False),
           fit_params=None, iid='warn', n_jobs=4,
           param_grid={'max_depth': [2, 4, 8, 15], 'max_features': ['auto', 'sqrt']},
           pre_dispatch='2*n_jobs', refit=True, return_train_score=True,
           scoring='roc_auc', verbose=0)


