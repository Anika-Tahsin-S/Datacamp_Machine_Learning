model = RandomForestRegressor(n_estimator = 500, random_state = 1111)
model.fit(X = X_train, y = y_train)
predictions = model.predict(X_test)
print("{0:.2f}".format(mae(y_true = y_test, y_pred = predictions)))




##                   Modeling steps                  ##
# Which of the following is NOT a valid method in the four-step scikit-learn model validation framework?
# Answer: .validate()
# Validation is a technique all in its own and is not completed with .validate(). You need to learn a few tools and techniques before you can validate a model.



##                   Seen vs. unseen data                  ##
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

# output:<br>
#     Model error on seen data: 3.28.
#     Model error on unseen data: 11.06.
# 
# When models perform differently on training and testing data, you should look to model validation to ensure you have the best performing model. 