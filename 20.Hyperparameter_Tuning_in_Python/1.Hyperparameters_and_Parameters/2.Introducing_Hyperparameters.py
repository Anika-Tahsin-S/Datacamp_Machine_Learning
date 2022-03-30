## Some important hyperparameters:
# n_estimators (high values)
# max_features (try different values)
# max_depth & min_sample_leaf (important for overfiting)
# criterion (maybe)




# --------------------------------------------------------------------------------------------------------- #
##                  Hyperparameters in Random Forests                  ##
print(RandomForestClassifier())
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators='warn', n_jobs=None,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)

# Which of the following is a hyperparameter for the Scikit Learn random forest model?
# Answer: oob_score
# oob_score set to True or False decides whether to use out-of-bag samples to estimate the generalization accuracy.





##                  Exploring Random Forest Hyperparameters                  ##
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Part 1
# Print out the old estimator, notice which hyperparameter is badly set
print(rf_clf_old)

# Get confusion matrix & accuracy for the old rf_model
print("Confusion Matrix: \n\n {} \n Accuracy Score: \n\n {}".format(
  	confusion_matrix(y_test, rf_old_predictions),
  	accuracy_score(y_test, rf_old_predictions))) 

# output:
#  RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#                 max_depth=None, max_features='auto', max_leaf_nodes=None,
#                 min_impurity_decrease=0.0, min_impurity_split=None,
#                 min_samples_leaf=1, min_samples_split=2,
#                 min_weight_fraction_leaf=0.0, n_estimators=5, n_jobs=None,
#                 oob_score=False, random_state=42, verbose=0, warm_start=False)
#     Confusion Matrix:
#      [[276  37]
#      [ 64  23]] 
#      Accuracy Score: 0.7475

# Part 2
# Create a new random forest classifier with better hyperparamaters
rf_clf_new = RandomForestClassifier(n_estimators = 500)

# Fit this to the data and obtain predictions
rf_new_predictions = rf_clf_new.fit(X_train, y_train).predict(X_test)

# Part 3
# Assess the new model (using new predictions!)
print("Confusion Matrix: \n\n", confusion_matrix(y_test, rf_new_predictions))
print("Accuracy Score: \n\n", accuracy_score(y_test, rf_new_predictions))

# output:
#     Confusion Matrix:   
#      [[300  13]
#      [ 63  24]]
#     Accuracy Score: 0.81

# We got a nice 5% accuracy boost just from changing the n_estimators. 






##                  Hyperparameters of KNN                  ##
# Build a knn estimator for each value of n_neighbours
knn_5 = KNeighborsClassifier(n_neighbors = 5)
knn_10 = KNeighborsClassifier(n_neighbors = 10)
knn_20 = KNeighborsClassifier(n_neighbors = 20)

# Fit each to the training data & produce predictions
knn_5_predictions = knn_5.fit(X_train, y_train).predict(X_test)
knn_10_predictions = knn_10.fit(X_train, y_train).predict(X_test)
knn_20_predictions = knn_20.fit(X_train, y_train).predict(X_test)

# Get an accuracy score for each of the models
knn_5_accuracy = accuracy_score(y_test, knn_5_predictions)
knn_10_accuracy = accuracy_score(y_test, knn_10_predictions)
knn_20_accuracy = accuracy_score(y_test, knn_20_predictions)
print("The accuracy of 5, 10, 20 neighbours was {}, {}, {}".format(knn_5_accuracy, knn_10_accuracy, knn_20_accuracy))

# output: The accuracy of 5, 10, 20 neighbours was 0.7125, 0.765, 0.7825

