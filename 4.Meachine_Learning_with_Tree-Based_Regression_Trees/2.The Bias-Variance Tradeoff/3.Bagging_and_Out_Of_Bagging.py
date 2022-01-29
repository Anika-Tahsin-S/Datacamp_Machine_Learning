# Import models and utility functions
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier

# Set seed for reproducibility
SEED = 1

# Split data into 70% train and 30% test
X_train, y_train, X_test, y_test = train_test_split(X, y, test_size = 0.3, stratify = y, random_state = SEED)


##                  Define The Bagging Classifier                  ##
# The following exercises is on the Indian Liver Patient dataset from the UCI machine learning repository. 

# Instantiate a classification-tree 'dt'
dt = DecisionTreeClassifier(max_depth = 4, samples_leaf = 0.16, random_state = SEED)

# Instantiate a BaggingClassifier 'bc'
# Set the paramter n_jobs to -1 so that all CPU cores are used in computation.
bc = BaggingClassifier(base_estimator = dt, n_estimators = 50, n_jobs = -1)

##                  Evaluate Bagging performance                  ##
# Fit 'bc' to the training set
bc.fit(X_train, y_train)

# Predict test set labels
y_pred = bc.predict(X_test)

# Evaluate and print test-set accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy of Bagging Classifier: {:.3f}'.format(accuracy))



##                  Out Of Bag Evaluation                  ##
# Instantiate a BaggingClassifier 'bc'
# Set the parameter oob_score to True in order to evaluate the OOB-accuracy of bc after training. 
# Note: In scikit-learn, the OOB-score corresponds to the accuracy for classifiers and the r-squared score for regressors
bc = BaggingClassifier(base_estimator = dt, n_estimators = 50, oob_score = True, n_jobs = -1)

# Fit 'bc' to the training set
bc.fit(X_train, y_train)

# Predict test set labels
y_pred = bc.predict(X_test)

# Evaluate test-set accuracy
test_accuracy = accuracy_score(y_test, y_pred)

# Extract the OOB accuracy from 'bc'
oob_accuracy = bc.oob_score_

# Print test set and OOB accuracy
print('Test Set Accuracy: {:.3f}'.format(test_accuracy))
print('OOB Accuracy: {:.3f}'.format(oob_accuracy))