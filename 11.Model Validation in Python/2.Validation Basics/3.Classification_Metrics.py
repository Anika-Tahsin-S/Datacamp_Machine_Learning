##                   Confusion Matrices                  ##
# Calculate and print the accuracy
accuracy = (491 + 324) / (953)
print("The overall accuracy is {0: 0.2f}".format(accuracy))

# Calculate and print the precision
precision = (491) / (491 + 15)
print("The precision is {0: 0.2f}".format(precision))

# Calculate and print the recall
recall = (491) / (123 + 491)
print("The recall is {0: 0.2f}".format(recall))
# output:
#     The overall accuracy is  0.86
#     The precision is  0.97
#     The recall is  0.80





##                   Confusion Matrices, Again                  ##
from sklearn.metrics import confusion_matrix

# Create predictions
test_predictions = rfc.predict(X_test)

# Create and print the confusion matrix
cm = confusion_matrix(y_test, test_predictions)
print(cm)

# Print the true positives (actual 1s that were predicted 1s)
print("The number of true positives is: {}".format(cm[1, 1]))
# output:
#     [[177 123]
#      [ 92 471]]
#     The number of true positives is: 471





##                   Precision vs. Recall                  ##
from sklearn.metrics import precision_score

test_predictions = rfc.predict(X_test)

# Create precision or recall score based on the metric you imported
score = precision_score(y_test, test_predictions)

# Print the final result
print("The precision value is {0:.2f}".format(score))
# output:
#     The precision value is 0.79