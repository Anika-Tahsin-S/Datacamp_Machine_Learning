##                  Using the best outputs                  ##
# Which of the following parameters must be set in order to be able to directly use the best_estimator_ property for predictions?
# Answer: refit = True
# When we set this to true, the creation of the grid search object automatically refits the best parameters on the whole training set and creates the best_estimator_ property.




##                  Exploring the grid search results                  ##
# Read the cv_results property into a dataframe & print it out
cv_results_df = pd.DataFrame(grid_rf_class.cv_results_)
print(cv_results_df)

# Extract and print the column with a dictionary of hyperparameters used
column = cv_results_df.loc[:, ['params']]
print(column)

# Extract and print the row that had the best mean test score
best_row = cv_results_df[cv_results_df['rank_test_score'] == 1 ]
print(best_row)

# output:
#         mean_fit_time  std_fit_time  mean_score_time  std_score_time param_max_depth param_min_samples_leaf param_n_estimators                                             params  split0_test_score       ...         std_test_score  rank_test_score  split0_train_score  split1_train_score  split2_train_score  split3_train_score  split4_train_score  mean_train_score  std_train_score
#     0        0.324582      0.004639         0.020115        0.003147              10                      1                100  {'max_depth': 10, 'min_samples_leaf': 1, 'n_es...           0.728204       ...               0.029951                9            0.994760            0.995460            0.995433            0.998330            0.996505          0.996098     1.248173e-03
#     1        0.671539      0.018667         0.039659        0.006038              10                      1                200  {'max_depth': 10, 'min_samples_leaf': 1, 'n_es...           0.735397       ...               0.027214                4            0.996087            0.995598            0.996509            0.999351            0.997369          0.996983     1.319340e-03
#     2        0.977654      0.011564         0.054303        0.003766              10                      1                300  {'max_depth': 10, 'min_samples_leaf': 1, 'n_es...           0.729294       ...               0.027756                1            0.996046            0.996136            0.996882            0.999655            0.997122          0.997168     1.311001e-03
#     3        0.313844      0.008520         0.017680        0.002485              10                      2                100  {'max_depth': 10, 'min_samples_leaf': 2, 'n_es...           0.728204       ...               0.027946                3            0.988662            0.984090            0.991403            0.994205            0.990968          0.989866     3.382634e-03
#     4        0.645001      0.003995         0.034633        0.004523              10                      2                200  {'max_depth': 10, 'min_samples_leaf': 2, 'n_es...           0.728858       ...               0.032881                2            0.988980            0.990355            0.993321            0.994922            0.992832          0.992082     2.134008e-03
#     5        0.967691      0.010192         0.057124        0.003817              10                      2                300  {'max_depth': 10, 'min_samples_leaf': 2, 'n_es...           0.721665       ...               0.033275                6            0.988717            0.990755            0.992756            0.995088            0.992531          0.991969     2.131063e-03
#     6        0.342355      0.002576         0.020001        0.002485              20                      1                100  {'max_depth': 20, 'min_samples_leaf': 1, 'n_es...           0.720684       ...               0.022226               11            1.000000            1.000000            1.000000            1.000000            1.000000          1.000000     4.965068e-17
#     7        0.724901      0.003929         0.034221        0.002700              20                      1                200  {'max_depth': 20, 'min_samples_leaf': 1, 'n_es...           0.725262       ...               0.030262               10            1.000000            1.000000            1.000000            1.000000            1.000000          1.000000     4.965068e-17
#     8        1.078176      0.013817         0.056005        0.002604              20                      1                300  {'max_depth': 20, 'min_samples_leaf': 1, 'n_es...           0.732672       ...               0.024880                5            1.000000            1.000000            1.000000            1.000000            1.000000          1.000000     0.000000e+00
#     9        0.351497      0.010176         0.020151        0.004323              20                      2                100  {'max_depth': 20, 'min_samples_leaf': 2, 'n_es...           0.718178       ...               0.025444               12            0.997567            0.999379            0.998289            0.999724            0.998383          0.998668     7.821910e-04
#     10       0.695131      0.018459         0.040476        0.001271              20                      2                200  {'max_depth': 20, 'min_samples_leaf': 2, 'n_es...           0.726896       ...               0.021853                8            0.998728            0.999089            0.998579            0.999710            0.998904          0.999002     3.932301e-04
#     11       0.792999      0.021011         0.030246        0.003575              20                      2                300  {'max_depth': 20, 'min_samples_leaf': 2, 'n_es...           0.730384       ...               0.020686                7            0.997539            0.998979            0.998938            0.999862            0.998904          0.998844     7.443455e-04
#     
#     [12 rows x 23 columns]
#                                                    params
#     0   {'max_depth': 10, 'min_samples_leaf': 1, 'n_es...
#     1   {'max_depth': 10, 'min_samples_leaf': 1, 'n_es...
#     2   {'max_depth': 10, 'min_samples_leaf': 1, 'n_es...
#     3   {'max_depth': 10, 'min_samples_leaf': 2, 'n_es...
#     4   {'max_depth': 10, 'min_samples_leaf': 2, 'n_es...
#     5   {'max_depth': 10, 'min_samples_leaf': 2, 'n_es...
#     6   {'max_depth': 20, 'min_samples_leaf': 1, 'n_es...
#     7   {'max_depth': 20, 'min_samples_leaf': 1, 'n_es...
#     8   {'max_depth': 20, 'min_samples_leaf': 1, 'n_es...
#     9   {'max_depth': 20, 'min_samples_leaf': 2, 'n_es...
#     10  {'max_depth': 20, 'min_samples_leaf': 2, 'n_es...
#     11  {'max_depth': 20, 'min_samples_leaf': 2, 'n_es...
#        mean_fit_time  std_fit_time  mean_score_time  std_score_time param_max_depth param_min_samples_leaf param_n_estimators                                             params  split0_test_score       ...         std_test_score  rank_test_score  split0_train_score  split1_train_score  split2_train_score  split3_train_score  split4_train_score  mean_train_score  std_train_score
#     2       0.977654      0.011564         0.054303        0.003766              10                      1                300  {'max_depth': 10, 'min_samples_leaf': 1, 'n_es...           0.729294       ...               0.027756                1            0.996046            0.996136            0.996882            0.999655            0.997122          0.997168         0.001311
#     
#     [1 rows x 23 columns]









##                  Analyzing the best results                  ##
# Print out the ROC_AUC score from the best-performing square
best_score = grid_rf_class.best_score_
print(best_score)

# Create a variable from the row related to the best-performing square
cv_results_df = pd.DataFrame(grid_rf_class.cv_results_)
best_row = cv_results_df.loc[[grid_rf_class.best_index_]]
print(best_row)

# Get the n_estimators parameter from the best-performing square and print
best_n_estimators = grid_rf_class.best_params_["n_estimators"]
print(best_n_estimators)

# output:
#     0.7718143027468853
#        mean_fit_time  std_fit_time  mean_score_time  std_score_time param_max_depth param_min_samples_leaf param_n_estimators                                             params  split0_test_score       ...         std_test_score  rank_test_score  split0_train_score  split1_train_score  split2_train_score  split3_train_score  split4_train_score  mean_train_score  std_train_score
#     2       0.977654      0.011564         0.054303        0.003766              10                      1                300  {'max_depth': 10, 'min_samples_leaf': 1, 'n_es...           0.729294       ...               0.027756                1            0.996046            0.996136            0.996882            0.999655            0.997122          0.997168         0.001311
#     
#     [1 rows x 23 columns]
#     300





##                  Using the best results                  ##
# See what type of object the best_estimator_ property is
print(type(grid_rf_class.best_estimator_))

# Create an array of predictions directly using the best_estimator_ property
predictions = grid_rf_class.best_estimator_.predict(X_test)

# Take a look to confirm it worked, this should be an array of 1's and 0's
print(predictions[0:5])

# Now create a confusion matrix 
print("Confusion Matrix \n", confusion_matrix(y_test, predictions))

# Get the ROC-AUC score
predictions_proba = grid_rf_class.best_estimator_.predict_proba(X_test)[:,1]
print("ROC-AUC Score \n", roc_auc_score(y_test, predictions_proba))

# output:
#     <class 'sklearn.ensemble.forest.RandomForestClassifier'>
#     [0 0 0 0 1]
#     Confusion Matrix 
#      [[140   8]
#      [ 36  16]]
#     ROC-AUC Score 
#      0.7436330561330562