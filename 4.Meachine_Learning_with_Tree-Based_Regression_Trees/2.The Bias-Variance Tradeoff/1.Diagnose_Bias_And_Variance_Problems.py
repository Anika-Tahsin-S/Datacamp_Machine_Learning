from sklearn.tree import DecisionTreeRegressor
# Import train_test_split from sklearn.model_selection
from sklearn.model_selection import train_test_split
# Import mean_squared_error from sklearn.metrics as MSE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import cross_val_score

##                  Instantiate The Model                  ##

# Set SEED for reproducibility
SEED = 1

# Split the data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=SEED)

# Instantiate a DecisionTreeRegressor dt
dt = DecisionTreeRegressor(max_depth = 4, min_samples_leaf = 0.26, random_state=SEED)

##                  Evaluating The 10-Fold Error                  ##

# Compute the array containing the 10-folds CV MSEs
# Set n_jobs to -1 to exploit all available CPUs in computation. 
MSE_CV_scores = - cross_val_score(dt, X_train, y_train, cv = 10, 
                       scoring = 'neg_mean_squared_error',
                       n_jobs = -1)

# Compute the 10-folds CV RMSE
RMSE_CV = (MSE_CV_scores.mean())**(1/2)

# Print RMSE_CV
print('CV RMSE: {:.2f}'.format(RMSE_CV))


##                  Evaluating Training Error                  ##

# Fit dt to the training set
dt.fit(X_train, y_train)

# Predict the labels of the training set
y_pred_train = dt.predict(X_train)

# Evaluate the training set RMSE of dt
RMSE_train = (MSE(y_train, y_pred_train))**(1/2)

# Print RMSE_train
print('Train RMSE: {:.2f}'.format(RMSE_train))

# High Bias or High Variance?
# RMSE_CV = 5.14, RMSE_train = 5.15, baseline_RMSE = 5.1
# RMSE_CV < RMSE_train means dt suffers from high bias because RMSE_CV ≈ RMSE_train and both scores are greater than baseline_RMSE.
# dt is indeed underfitting the training set as the model is too constrained to capture the nonlinear dependencies between features and labels.