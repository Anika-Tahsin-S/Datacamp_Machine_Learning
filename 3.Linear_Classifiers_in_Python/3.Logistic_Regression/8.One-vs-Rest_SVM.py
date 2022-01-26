# The data is loaded into X_train, y_train, X_test, and y_test .
# Instead of using LinearSVC, we'll now use scikit-learn's SVC object, which is a non-linear "kernel" SVM

# We'll use SVC instead of LinearSVC from now on
from sklearn.svm import SVC

# Create/plot the binary classifier (class 1 vs. rest)
svm_class_1 = SVC()
svm_class_1.fit(X_train, y_train == 1) # y_train == 1, is ovr Strategy
plot_classifier(X_train, y_train == 1, svm_class_1)