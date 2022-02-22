import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

churn_data = pd.read_csv("classification_data.csv")

X, y = churn_data.iloc[:,:-1], churn_data.iloc[:,-1] # split the entire dataset into a matrix of samples by features, called X by convention, and a vector of target values, called y by convention.

X_train, y_train, X_test, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)
xg_cl =xgb.XGBClassifier(objective = 'binary:logistic', n_estimators = 10, seed = 123)
xg_cl.fit(X_train, y_train)
preds = xg_cl.predict(X_test)

accuracy = float(np.sum(preds == y_test))/y_test.shape[0]

print("accuarcy: %f" % (accuracy))



##                   Decision trees                  ##
# Import the necessary modules
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Create the training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Instantiate the classifier: dt_clf_4
dt_clf_4 = DecisionTreeClassifier(max_depth = 4)

# Fit the classifier to the training set
dt_clf_4.fit(X_train, y_train)

# Predict the labels of the test set: y_pred_4
y_pred_4 = dt_clf_4.predict(X_test)

# Compute the accuracy of the predictions: accuracy
accuracy = float(np.sum(y_pred_4==y_test))/y_test.shape[0]
print("accuracy:", accuracy)