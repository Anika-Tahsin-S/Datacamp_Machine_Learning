##                   Create one Holdout Set                  ##
# Create dummy variables using pandas
X = pd.get_dummies(tic_tac_toe.iloc[:,0:9])
y = tic_tac_toe.iloc[:, 9]

# Create training and testing datasets. Use 10% for the test set
X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size = 0.10, random_state = 1111)



##                   Create two Holdout Set                  ##
# Create temporary training and final testing datasets
X_temp, X_test, y_temp, y_test  =\
    train_test_split(X, y, test_size = 0.2, random_state = 1111)

# Create the final training and validation datasets
X_train, X_val, y_train, y_val =\
    train_test_split(X_temp, y_temp, test_size = 0.25, random_state = 1111)



##                   Why use Holdout Sets                  ##
# When should you consider using training, validation, and testing datasets?
# Answer: When testing parameters, tuning hyper-parameters, or anytime you are frequently evaluating model performance.

# Anytime we are evaluating model performance repeatedly we need to create training, validation, and testing datasets.