import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


## Parameters in Logistic Regression
# Log Reg model
log_reg_clf = LogisticRegression()
log_reg_clf.fit(X_train, y_train)
print(log_reg_clf.coef_)

# Tidy up coef
original_variables = list(X_train.columns)

zip_together = list(zip(original_variables, log_reg_clf.coef_[0]))

coefs = [list(x) for x in zipped_together]
coefs = pd.DataFrame(coefs, columns = ['Variable', 'Coefficient'])

# sort
coefs.sort_values(by = ['Coefficient'], axis = 0, inplace = True, ascending = False)
print(coefs.head(3))



## Parameters in Random Forest
rf_clf = RandomForestClassifier(max_depth = 2)
rf_clf.fit(X_train, y_train)
# Pull out one tree from the forest
chosen_tree = rf_clf.estimators_[7]


# Extracting Node Decisions
split_column = chosen_tree.tree_.feature[1]
split_column_name = x_train.columns[split_column]

split_value = chosen_tree.tree_.threshold[1]
print('This node split on feature {}, at a value pf {}'.format(split_column_name, split_value))








X = credit_card.drop(['ID', 'default payment next month'], axis = 1)
y = credit_card['default payment next month']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, shuffle = True)


# --------------------------------------------------------------------------------------------------------- #
##                  Parameters in Logistic Regression                  ##
# Which of the following is a parameter for the Scikit Learn logistic regression model? Here we mean conceptually based on the theory introduced in this course. NOT what the Scikit Learn documentation calls a parameter or attribute.
# Answer: coef_
# coef_ contains the important information about coefficients on our variables in the model. We do not set this, it is learned by the algorithm through the modeling process.
# LogisticRegression() : estimator





##                  Extracting a Logistic Regression parameter                  ##
# Create a list of original variable names from the training DataFrame
original_variables = list(X_train.columns)

# Extract the coefficients of the logistic regression estimator
model_coefficients = log_reg_clf.coef_[0]

# Create a dataframe of the variables and coefficients & print it out
coefficient_df = pd.DataFrame({"Variable" : original_variables, "Coefficient": model_coefficients})
print(coefficient_df)

# Print out the top 3 positive variables
top_three_df = coefficient_df.sort_values(by = ['Coefficient'], axis = 0, ascending = False)[0:3]
print(top_three_df)

# output:
#            Variable   Coefficient
#     0     LIMIT_BAL -2.886513e-06
#     1           AGE -8.231685e-03
#     2         PAY_0  7.508570e-04
#     3         PAY_2  3.943751e-04
#     4         PAY_3  3.794236e-04
#     5         PAY_4  4.346120e-04
#     6         PAY_5  4.375615e-04
#     7         PAY_6  4.121071e-04
#     8     BILL_AMT1 -6.410891e-06
#     9     BILL_AMT2 -4.393645e-06
#     10    BILL_AMT3  5.147052e-06
#     11    BILL_AMT4  1.476978e-05
#     12    BILL_AMT5  2.644462e-06
#     13    BILL_AMT6 -2.446051e-06
#     14     PAY_AMT1 -5.448954e-05
#     15     PAY_AMT2 -8.516338e-05
#     16     PAY_AMT3 -4.732779e-05
#     17     PAY_AMT4 -3.238528e-05
#     18     PAY_AMT5 -3.141833e-05
#     19     PAY_AMT6  2.447717e-06
#     20        SEX_2 -2.240863e-04
#     21  EDUCATION_1 -1.642599e-05
#     22  EDUCATION_2 -1.777295e-04
#     23  EDUCATION_3 -5.875596e-05
#     24  EDUCATION_4 -3.681278e-06
#    25  EDUCATION_5 -7.865964e-06
#     26  EDUCATION_6 -9.450362e-07
#     27   MARRIAGE_1 -5.036826e-05
#     28   MARRIAGE_2 -2.254362e-04
#     29   MARRIAGE_3  1.070545e-05
#       Variable  Coefficient
#     2    PAY_0     0.000751
#     6    PAY_5     0.000438
#     5    PAY_4     0.000435







##                  Extracting a Random Forest parameter                  ##
# Extract the 7th (index 6) tree from the random forest
chosen_tree = rf_clf.estimators_[6]

# Visualize the graph using the provided image
imgplot = plt.imshow(tree_viz_image)
plt.show()

# Extract the parameters and level of the top (index 0) node
split_column = chosen_tree.tree_.feature[0]
split_column_name = X_train.columns[split_column]
split_value = chosen_tree.tree_.threshold[0]

# Print out the feature and level
print("This node split on feature {}, at a value of {}".format(split_column_name, split_value))

# output: This node split on feature PAY_4, at a value of 1.0
