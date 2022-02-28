##                   Mean Absolute Error                  ##
from sklearn.metrics import mean_absolute_error

# Manually calculate the MAE
n = len(predictions)
mae_one = sum(abs(y_test - predictions)) / n
print('With a manual calculation, the error is {}'.format(mae_one))

# Use scikit-learn to calculate the MAE
mae_two = mean_absolute_error(y_test, predictions)
print('Using scikit-learn, the error is {}'.format(mae_two))
# Output: 
#   With a manual calculation, the error is 5.9
#   Using scikit-learn, the error is 5.9




##                   Mean Squared Error                  ##
from sklearn.metrics import mean_squared_error

n = len(predictions)
# Finish the manual calculation of the MSE
mse_one = sum((y_test - predictions) ** 2) / n
print('With a manual calculation, the error is {}'.format(mse_one))

# Use the scikit-learn function to calculate MSE
mse_two = mean_squared_error(y_test, predictions)
print('Using scikit-learn, the error is {}'.format(mse_two))
# output:
#     With a manual calculation, the error is 49.1
#     Using scikit-learn, the error is 49.1






##                   Performance on Data Subsets                  ##
# Part 1
# Find the East conference teams
east_teams = labels == 'E'

# Part 2
# Create arrays for the true and predicted values
true_east = y_test[east_teams]
preds_east = predictions[east_teams]

# Part 3
# Print the accuracy metrics
print('The MAE for East teams is {}'.format(
    mae(true_east, preds_east)))
# output: The MAE for East teams is 6.733333333333333


# Part 4
# Print the West accuracy
print('The MAE for West conference is {}'.format(west_error))
# Output :
#     The MAE for East teams is 6.733333333333333
#     The MAE for West conference is 5.01