# dot-get_params() is used to review which parameters went into a scikit-learn model. 
# It will print out a dictionary of parameters and their values, allowing us to see exactly which parameters were used.

# dot-score() is a quick way to look at the overall accuracy of the classification model.




##                   Classification Predictions                  ##
# Fit the rfc model. 
rfc.fit(X_train, y_train)

# Create arrays of predictions
classification_predictions = rfc.predict(X_test)
probability_predictions = rfc.predict_proba(X_test)

# Print out count of binary predictions
print(pd.Series(classification_predictions).value_counts())

# Print the first value from probability_predictions
print('The first predicted probabilities are: {}'.format(probability_predictions[0]))
# output:
#     1    563
#     0    204
#     dtype: int64
#     The first predicted probabilities are: [0.26524423 0.73475577]






##                   Reusing Model Parameters                  ##
rfc = RandomForestClassifier(n_estimators = 50, max_depth = 6, random_state = 1111)

# Print the classification model
print(rfc)

# Print the classification model's random state parameter
print('The random state is: {}'.format(rfc.random_state))

# Print all parameters
print('Printing the parameters dictionary: {}'.format(rfc.get_params()))
# output: RandomForestClassifier(max_depth=6, n_estimators=50, random_state=1111)
# The random state is: 1111
# Printing the parameters dictionary: {'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini', 'max_depth': 6, 'max_features': 'auto', 
# 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 
# 'n_estimators': 50, 'n_jobs': None, 'oob_score': False, 'random_state': 1111, 'verbose': 0, 'warm_start': False}




##                   Random Forest Classifier                  ##
# Part 1
from sklearn.ensemble import RandomForestClassifier

# Create a random forest classifier
rfc = RandomForestClassifier(n_estimators = 50, max_depth = 6, random_state = 1111)


# Part 2
# Fit rfc using X_train and y_train
rfc.fit(X_train, y_train)


# Part 3
# Create predictions on X_test
predictions = rfc.predict(X_test)
print(predictions[0:5])
# output: [1 1 1 1 1]


# Part 4
# Print model accuracy using score() and the testing data
print(rfc.score(X_test, y_test))
# Output: 0.817470664928292