# The features and targets are already loaded for you in X_train and y_train.
# We'll search for the best value of C using scikit-learn's GridSearchCV(), which was covered in the prerequisite course.

# Specify L1 regularization
lr = LogisticRegression('l1')

# Instantiate the GridSearchCV object and run the search
searcher = GridSearchCV(lr, {'C':[0.001, 0.01, 0.1, 1, 10]})
searcher.fit(X_train, y_train)

# Report the best parameters
print("Best CV params", searcher.best_params_)

# Find the number of nonzero coefficients (selected features)
best_lr = searcher.best_estimator_
coefs = best_lr.coef_
print("Total number of features:", coefs.size)
print("Number of selected features:", np.count_nonzero(coefs))