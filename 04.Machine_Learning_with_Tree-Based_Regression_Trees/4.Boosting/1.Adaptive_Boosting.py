from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import AdaBoostClassifier

# Set seed for reproducibility
SEED = 1

# Split data into 80% train and 20% test
X_train, y_train, X_test, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = SEED)


##                   Define The AdaBoost Classifier                  ##
# Instantiate a classification-tree 'dt'
dt = DecisionTreeClassifier(max_depth = 2, random_state = SEED)

# Instantiate an AdaBoost Classifier 'adb_clf'
adb_clf = AdaBoostClassifier(base_estimator = dt, n_estimators = 180, random_state = SEED)



##                   Train The AdaBoost classifier                  ##
# Fit 'adb_clf' to the training set
adb_clf.fit(X_train, y_train)

# Predict the test set probabilities of positive class
y_pred_prob = adb_clf.predict_proba(X_test)[:,1]



##                   Evaluate The AdaBoost classifier                  ##
# Evaluate test-set roc_auc_score
adb_clf_roc_auc_score = roc_auc_score(y_test, y_pred_prob)

# Print adb_clf_roc_auc_score
print('ROC AUC score: {:.2f}'.format(adb_clf_roc_auc_score))