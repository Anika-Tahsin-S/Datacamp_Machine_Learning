##                   Best Classification Accuracy                  ##
from sklearn.ensemble import RandomForestRegressor

# Fill in rfr using your variables
rfr = RandomForestRegressor(
    n_estimators = 100,
    max_depth = random.max_depth(max_depth),
    min_samples_split = random.min_samples_split(min_samples_split),
    max_features = random.max_features(max_features))

# Print out the parameters
print(rfr.get_params())

print(rs.best_estimator_)
# Which parameter set produces the best classification accuracy?
# Answer: RandomForestClassifier(max_depth = 12, min_samples_split = 4, n_estimators = 20, random_state = 1111)





##                   Selecting the Best Precision Model                  ##
from sklearn.metrics import precision_score, make_scorer

# Create a precision scorer
precision = make_scorer(precision_score)
# Finalize the random search
rs = RandomizedSearchCV(
  estimator = rfc, param_distributions = param_dist,
  scoring = precision,
  cv = 5, n_iter = 10, random_state = 1111)
rs.fit(X, y)

# print the mean test scores:
print('The accuracy for each run was: {}.'.format(rs.cv_results_['mean_test_score']))
# print the best model score:
print('The best accuracy for a single model was: {}'.format(rs.best_score_))
# output:
#     The accuracy for each run was: [0.87614978 0.75561877 0.67740077 0.89141614 0.87024051 0.85772772
#      0.68244199 0.82867397 0.88717239 0.91980724].
#     The best accuracy for a single model was: 0.9198072369317106

# The model precision was 93%! The best model accurately predicts a winning game 93% of the time. 
# If you look at the mean test scores, you can tell some of the other parameter sets did really poorly. 
# Also, since you used cross-validation, you can be confident in your predictions.