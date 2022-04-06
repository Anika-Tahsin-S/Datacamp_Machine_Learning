## Data Type:
# Tabular data, time series, images, text, etc

## Problem Type:
# Classification, Regression, Ranking, etc

## Evaluation Metric:
# ROC, AUC, F1-Score, MAE, MSE, etc




## Metric Definition
from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error

# However, there are some special competition metrics that are not available in scikit-learn.
# In such cases, we have to create metrics manually. 
# Suppose we're solving the competition problem with Root Mean Squared Logarithmic Error as an evaluation metric. This metric is not implemented in scikit-learn.
# So, it is a usual Root Mean Squared Error in a logarithmic scale. In this situation, we have to define a custom function that takes as input the true and predicted values, and outputs the metric value.

import numpy as np

def rmsle(y_true, y_pred):
    diffs = np.log(y_true + 1) - np.log(y_pred + 1)
    squares = np.power(diffs, 2)
    err = np.sqrt(np.mean(squares))
    return err



## ====================================================================================================== ##