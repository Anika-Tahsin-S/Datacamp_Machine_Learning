from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Create the function
def gbm_grid_search(learn_rate, max_depth):
    # Create the model
    model = GradientBoostingClassifier(learning_rate = learn_rate, 
                                        max_depth = max_depth)
    
    # Use the model to make predictions
    predictions = model.fit(X_train, y_train).predict(X_test)
    
    # Return the hyperparameters and score
    return ([learn_rate, max_depth, accuracy_score(y_test, predictions)])

# nested loop
results_list = []

for learn_rate in learn_rate_list:
    for max_depth in max_depth_list:
        results_list.append(gbm_grid_search(learn_rate, max_depth))
        
results_df = pd.DataFrame(results_list, columns = ['learning_rate', 'max_depth', 'accuracy'])
print(results_df)


# Adding more hyperparameters
# Adjusting function
def gbm_grid_search_extended(learn_rate, max_depth, subsample, max_features):
    # Extend the model creation section
    model = GradientBoostingClassifier(learning_rate = learn_rate, 
                                       max_depth = max_depth,
                                       subsample = subsample,
                                       max_features = max_features)
    
    predictions = model.fit(X_train, y_train).predict(X_test)
    return([learn_rate, max_depth, subsample, accuracy_score(y_test, predictions)])

# Adjusting for loop
for learn_rate in learn_rate_list:
    for max_depth in max_depth_list:
        for subsample in subsample_list:
            for max_features in max_features_list:
                results_list.append(gbm_grid_search_extended(learn_rate, max_depth, subsample, max_features))
            
results_df = pd.DataFrame(results_list, columns = ['learning_rate', 'max_depth', 'accuracy', 'max_features', 'subsample'])
print(results_df)


# Intro to Grid Search
GradientBoostingClassifier(max_depth = 4, learning_rate = 0.001)









# --------------------------------------------------------------------------------------------------------- #
##                  Build Grid Search functions                  ##
# Create the function
def gbm_grid_search(learning_rate, max_depth):

	# Create the model
    model = GradientBoostingClassifier(learning_rate = learning_rate, max_depth = max_depth)
    
    # Use the model to make predictions
    predictions = model.fit(X_train, y_train).predict(X_test)
    
    # Return the hyperparameters and score
    return([learning_rate, max_depth, accuracy_score(y_test, predictions)])





##                  Iteratively tune multiple hyperparameters                  ##
# Part 1
# Create the relevant lists
results_list = []
learn_rate_list = [0.01, 0.1, 0.5]
max_depth_list = [2, 4, 6]

# Create the for loop
for learn_rate in learn_rate_list:
    for max_depth in max_depth_list:
        results_list.append(gbm_grid_search(learn_rate,max_depth))

# Print the results
print(results_list)   

# output: [[0.01, 2, 0.78], [0.01, 4, 0.78], [0.01, 6, 0.76], [0.1, 2, 0.74], [0.1, 4, 0.76], [0.1, 6, 0.75], [0.5, 2, 0.73], [0.5, 4, 0.74], [0.5, 6, 0.74]]

# Part 2
results_list = []
learn_rate_list = [0.01, 0.1, 0.5]
max_depth_list = [2,4,6]

# Extend the function input
def gbm_grid_search_extended(learn_rate, max_depth, subsample):

	# Extend the model creation section
    model = GradientBoostingClassifier(learning_rate = learn_rate, max_depth = max_depth, subsample = subsample)
    
    predictions = model.fit(X_train, y_train).predict(X_test)
    
    # Extend the return part
    return([learn_rate, max_depth, subsample, accuracy_score(y_test, predictions)])

# output: [[0.01, 2, 0.78], [0.01, 4, 0.78], [0.01, 6, 0.76], [0.1, 2, 0.74], [0.1, 4, 0.76], [0.1, 6, 0.75], [0.5, 2, 0.73], [0.5, 4, 0.74], [0.5, 6, 0.74]]


# Part 3
results_list = []

# Create the new list to test
subsample_list = [0.4 , 0.6]

for learn_rate in learn_rate_list:
    for max_depth in max_depth_list:
    
    	# Extend the for loop
        for subsample in subsample_list:
        	
            # Extend the results to include the new hyperparameter
            results_list.append(gbm_grid_search_extended(learn_rate, max_depth, subsample))
            
# Print results
print(results_list)            

# output: [[0.01, 2, 0.4, 0.73], [0.01, 2, 0.6, 0.74], [0.01, 4, 0.4, 0.73], [0.01, 4, 0.6, 0.75], [0.01, 6, 0.4, 0.72], [0.01, 6, 0.6, 0.78], [0.1, 2, 0.4, 0.74], [0.1, 2, 0.6, 0.74], [0.1, 4, 0.4, 0.73], [0.1, 4, 0.6, 0.73], [0.1, 6, 0.4, 0.74], [0.1, 6, 0.6, 0.76], [0.5, 2, 0.4, 0.64], [0.5, 2, 0.6, 0.67], [0.5, 4, 0.4, 0.72], [0.5, 4, 0.6, 0.71], [0.5, 6, 0.4, 0.63], [0.5, 6, 0.6, 0.64]]






##                  How Many Models?                  ##
# How many models would be created when running a grid search over the following hyperparameters and values for a GBM algorithm?
# 
# Answer: 1215
# For every value of one hyperparameter, we test EVERY value of EVERY other hyperparameter. So you correctly multiplied the number of values (the lengths of the lists).
