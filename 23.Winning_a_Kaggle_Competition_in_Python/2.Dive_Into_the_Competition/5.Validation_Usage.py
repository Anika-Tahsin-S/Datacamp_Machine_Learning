## Time K-fold cross-validation
from sklearn.model_selection import TimeSeriesSplit

time_kfold = TimeSeriesSplit(n_splits = 5)

# Sort train by date
train = train.sort_values('date')

for train_index, test_index in time_kfold.split(train):
    cv_train, cv_test = train.iloc[train_index], train.iloc[test_index]



## Validation Pipeline
fold_metrics = []

for train_index, test_index in CV_STRATEGY.split(train, train['target']):
    cv_train, cv_test = train.iloc[train_index], train.iloc[test_index]
    model.fit(cv_train)
    predictions = model.predict(cv_test)
    metric = evaluate(cv_test, predictions)
    fold_metrics.append(metric)



## Overall Validation Score
import numpy as np

# Simple mean over the folds
mean_score = np.mean(fold_metrics)

overall_score_minimizing = np.mean(fold_metrics) + np.std(fold_metrics)
overall_score_maximizing = np.mean(fold_metrics) - np.std(fold_metrics)




## ====================================================================================================== ##