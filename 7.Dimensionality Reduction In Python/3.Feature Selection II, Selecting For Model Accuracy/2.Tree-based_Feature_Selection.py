from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


##                   Automatic Recursive Feature Elimination                  ##
# Perform a 75% training and 25% test data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Fit the random forest model to the training data
rf = RandomForestClassifier(random_state = 0)
rf.fit(X_train, y_train)

# Calculate the accuracy
acc = accuracy_score(y_test, rf.predict(X_test))

# Print the importances per feature
print(dict(zip(X.columns, rf.feature_importances_.round(2))))

# Print accuracy
print("{0:.1%} accuracy on test set.".format(acc))
# {'pregnant': 0.09, 'glucose': 0.21, 'diastolic': 0.08, 'triceps': 0.11, 'insulin': 0.13, 'bmi': 0.09, 'family': 0.12, 'age': 0.16}
# 77.6% accuracy on test set.



##                   Random Forest For Feature Selection                  ##
# Create a mask for features importances above the threshold
mask = rf.feature_importances_ > 0.15

# Apply the mask to the feature dataset X
reduced_X = X.loc[:, mask]

# prints out the selected column names
print(reduced_X.columns)
# Only the features 'glucose' and 'age' were considered sufficiently important.




##                   Recursive Feature Elimination with Random Forest                  ##
# Pre-loaded datasets (Automatic Recursive Feature Elimination)
# Set the feature eliminator to remove 2 features on each step
rfe = RFE(estimator = RandomForestClassifier(n_estimators = 10), n_features_to_select = 2, step = 2, verbose = 1)

# Fit the model to the training data
rfe.fit(X_train, y_train)

# Create a mask
mask = rfe.support_

# Apply the mask to the feature dataset X and print the result
reduced_X = X.loc[:, mask]
print(reduced_X.columns)