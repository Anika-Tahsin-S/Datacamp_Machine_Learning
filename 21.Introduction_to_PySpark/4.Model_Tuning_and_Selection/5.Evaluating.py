##                  Evaluating binary classifiers                  ##
# For this course we'll be using a common metric for binary classification algorithms call the AUC, or area under the curve. In this case, the curve is the ROC, or receiver operating curve. The details of what these things actually measure isn't important for this course. All you need to know is that for our purposes, the closer the AUC is to one (1), the better the model is!

# If you've created a perfect binary classification model, what would the AUC be?

# Answer: 1; An AUC of one represents a model that always perfectly classifies observations.





##                  Evaluate the model                  ##
# Use the model to predict the test set
test_results = best_lr.transform(test)

# Evaluate the predictions
print(evaluator.evaluate(test_results))

# output: 0.7123313100891033