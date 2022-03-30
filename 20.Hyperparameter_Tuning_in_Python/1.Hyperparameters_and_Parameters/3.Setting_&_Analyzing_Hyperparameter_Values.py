# Automating Hyperparameter Tuning
from sklearn.ensemble import KNeighboraClassifier

neighbors_list = [3, 5, 10, 20, 50, 75]
for test_number in neighbors_list:
    model = KNeighboraClassifier(neighbors_list= test_number)
    predictions = model.fit(X_train, y_train).predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    accuracy_list.append(accuracy)

# Gather everything into a DataFrame
results_df = pd.DataFrame({'neighbors':neighbors_list, 'accuracy':accuracy_list})
print(results_df)

# Learning Curves
neighbors_list = list(range(5, 500, 5))
accuracy_list = []
for test_number in neighbors_list:
    model = KNeighboraClassifier(neighbors_list= test_number)
    predictions = model.fit(X_train, y_train).predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    accuracy_list.append(accuracy)

# Gather everything into a DataFrame
results_df = pd.DataFrame({'neighbors':neighbors_list, 'accuracy':accuracy_list})

# Plot DataFrame
plt.plot(results_df['neighbors'], results_df['accuracy'])
plt.gca().set(xlabel = 'n_neighbors', ylabel = 'Accuracy', title = 'Accuracy for different n_neighbors')
plt.show()

# Handy trick for generating values
print(np.linespace(1, 2, 5))








# --------------------------------------------------------------------------------------------------------- #
##                  Automating Hyperparameter Choice                  ##
# Set the learning rates & results storage
learning_rates = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
results_list = []

# Create the for loop to evaluate model predictions for each learning rate
for learning_rate in learning_rates:
    model = GradientBoostingClassifier(learning_rate = learning_rate)
    predictions = model.fit(X_train, y_train).predict(X_test)
    # Save the learning rate and accuracy score
    results_list.append([learning_rate, accuracy_score(y_test, predictions)])

# Gather everything into a DataFrame
results_df = pd.DataFrame(results_list, columns = ['learning_rate', 'accuracy'])
print(results_df)

# output:
#        learning_rate  accuracy
#     0          0.001    0.7825
#     1          0.010    0.8025
#     2          0.050    0.8100
#     3          0.100    0.7975
#     4          0.200    0.7900
#     5          0.500    0.7775





##                  Building Learning Curves                  ##
# Set the learning rates & accuracies list
learn_rates = np.linspace(0.01, 2, num = 30)
accuracies = []

# Create the for loop
for learn_rate in learn_rates:
  	# Create the model, predictions & save the accuracies as before
    model = GradientBoostingClassifier(learning_rate = learn_rate)
    predictions = model.fit(X_train, y_train).predict(X_test)
    accuracies.append(accuracy_score(y_test, predictions))

# Plot results    
plt.plot(learn_rates, accuracies)
plt.gca().set(xlabel = 'learning_rate', ylabel = 'Accuracy', title = 'Accuracy for different learning_rates')
plt.show()

