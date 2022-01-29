from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeCLassifier
from sklearn.metrics import roc_auc_score

# Set seed for reproducibility
SEED = 1

# Split data into 80% train and 20% test
X_train, y_train, X_test, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = SEED)

# Instantiate a DecisionTreeCLassifier 'dt'
dt = DecisionTreeCLassifier(random_state = SEED)


##                   Set The Tree's Hyperparameter Grid                  ##
# Define the grid of hyperparameters 'params_dt'
params_dt = {'max_depth' : [2, 3, 4],
            'min_samples_leaf' : [00.12, 0.14, 0.16, 0.18],
            'max_features' : [0.2, 0.4, 0.6, 0.8]}


##                   Search For The Optimal Tree                  ##
# Instantiate a 10-fold CV grid search object 'grid_dt'
grid_dt = GridSearchCV(estimator = dt, 
                        param_grid = params_dt,
                        scoring = 'ros_auc',
                        cv = 5,
                        n_jobs = - 1)

# Fit 'grid_dt' to the training set
grid_dt.fit(X_train, y_train)

# Extract best hyperparameters from 'grid_dt'
best_hyperparams = grid_dt.best_params_
print('Best hyperparameteres:\n', best_hyperparams)

# Extract best CV score from 'grid_dt'
best_CV_score = grid_dt.best_score_
print('Best CV accuracy'.format(best_CV_score))



##                   Evaluate The Optimal Tree                  ##
# Extract best model from 'grid_dt'
best_model = grid_dt.best_estimator_

# Evaluate test set accuracy
test_acc = best_model.score(X_test, y_test)
print('Test set accuracy of best model: {:.3f}'.format(test_acc))
# Print 'dt's hyperparameters
# Which prints out a dictionary where the keys are the hyperparameter names
print(dt.get_params()) 

# Predict the test set probabilities of the positive class
y_pred_proba = best_model.predict_proba(X_test)[:,1]

# Compute test_roc_auc
test_roc_auc = roc_auc_score(y_test, y_pred_proba)

# Print test_roc_auc
print('Test set ROC AUC score: {:.3f}'.format(test_roc_auc))