import xgboost as xgb
import pandas as pd

churn_data = pd.read_csv("classification_data.csv")


##                   Measuring accuracy                  ##
# We convert our dataset into an optimized data structure that the creators of XGBoost made that gives the package its lauded performance and efficiency gains called a DMatrix. In the previous exercise, the input datasets were converted into DMatrix data on the fly, but when we use the XGBoost cv object, which is part of XGBoost's learning api we have to first explicitly convert our data into a DMatrix. So, that's what we are doing here before we run our cross-validation.

# Create arrays for the features and the target: X, y
X, y = churn_data.iloc[:,:-1], churn_data.iloc[:,-1]

# Create the DMatrix from X and y: churn_dmatrix
churn_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary: params
params = {"objective":"reg:logistic", "max_depth":3}

# Perform cross-validation: cv_results
cv_results = xgb.cv(dtrain=churn_dmatrix, params=params, 
                  nfold=3, num_boost_round=5, 
                  metrics="error", as_pandas=True, seed=123)
# Print cv_results
print(cv_results)
# Print the accuracy
print(((1-cv_results["test-error-mean"]).iloc[-1]))
#print("Accuracy: %f" %((1-cv_results["test-error-mean"]).iloc[-1]))


##                   Measuring AUC                  ##

# Create the DMatrix: churn_dmatrix
churn_dmatrix = xgb.DMatrix(data = X, label = y)

# Create the parameter dictionary: params
params = {"objective" : "reg:logistic", "max_depth" : 3}

# Perform cross_validation: cv_results
cv_results = xgb.cv(dtrain = churn_dmatrix, params = params, 
                  nfold = 3, num_boost_round = 5, 
                  metrics = "auc", as_pandas = True, seed = 123)

# Print cv_results
print(cv_results)

# Print the AUC
print((cv_results["test-auc-mean"]).iloc[-1])

# An AUC of 0.84 is quite strong. As you have seen, XGBoost's learning API makes it very easy to compute any metric you may be interested in. In Chapter 3, you'll learn about techniques to fine-tune your XGBoost models to improve their performance even further. For now, it's time to learn a little about exactly when to use XGBoost.