from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

# n_estimators: the number of trees to create for the random forest.
# max_depth: the maximum depth for these trees, or how many times we can split the data.
# random_state allows us to create reproducible models. Can be used 1,111 as random state.
# After a model is created, we can assess how important different features (or columns) of the data were in the model by using the dot-feature_importances_ attribute.


##                   Set Parameters and Fit a Model                  ##
# Given in Shell #
# The model is fit using X_train and y_train
model.fit(X_train, y_train)

# Create vectors of predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Train/Test Errors
train_error = mae(y_true = y_train, y_pred = train_predictions)
test_error = mae(y_true = y_test, y_pred = test_predictions)

# Print the accuracy for seen and unseen data
print("Model error on seen data: {0:.2f}.".format(train_error))
print("Model error on unseen data: {0:.2f}.".format(test_error))
# -------------------------------------------------------------- #
# Set the number of trees
rfr.n_estimator = 100

# Add a maximum depth
rfr.max_depth = 6

# Set the random state
rfr.random_state = 1111

# Fit the model
rfr.fit(X_train, y_train)




##                   Feature Importances                  ##
# Fit the model using X and y
rfr.fit(X_train, y_train)

# Print how important each column is to the model
for i, item in enumerate(rfr.feature_importances_):
      # Use i and item to print out the feature importance of each column
    print("{0:s}: {1:.2f}".format(X_train.columns[i], item))
# output:
#     chocolate: 0.44
#     fruity: 0.03
#     caramel: 0.02
#     peanutyalmondy: 0.05
#     nougat: 0.01
#     crispedricewafer: 0.03
#     hard: 0.01
#     bar: 0.02
#     pluribus: 0.02
#     sugarpercent: 0.17
#     pricepercent: 0.19