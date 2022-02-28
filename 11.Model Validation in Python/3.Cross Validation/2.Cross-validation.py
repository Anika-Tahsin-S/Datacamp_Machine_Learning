##                   scikit-learn's KFold()                  ##
from sklearn.model_selection import KFold

# Use KFold
kf = KFold(n_splits = 5, shuffle = True, random_state = 1111)

# Create splits
splits = kf.split(X)

# Print the number of indices
for train_index, val_index in splits:
    print("Number of training indices: %s" % len(train_index))
    print("Number of validation indices: %s" % len(val_index))
# output:
#     Number of training indices: 68
#     Number of validation indices: 17
#     Number of training indices: 68
#     Number of validation indices: 17
#     Number of training indices: 68
#     Number of validation indices: 17
#     Number of training indices: 68
#     Number of validation indices: 17
#     Number of training indices: 68
#     Number of validation indices: 17




##                   Using KFold Indices                  ##
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

rfc = RandomForestRegressor(n_estimators = 25, random_state = 1111)

# Access the training and validation indices of splits
for train_index, val_index in splits:
    # Setup the training and validation data
    X_train, y_train = X[train_index], y[train_index]
    X_val, y_val = X[val_index], y[val_index]
    # Fit the random forest model
    rfc.fit(X_train, y_train)
    # Make predictions, and print the accuracy
    predictions = rfc.predict(X_val)
    print("Split accuracy: " + str(mean_squared_error(y_val, predictions)))
# output:
#     Split accuracy: 150.99298148707666
#     Split accuracy: 171.22206240542593
#     Split accuracy: 131.72569156195593
#     Split accuracy: 80.61940183841385
#     Split accuracy: 221.63020627476214