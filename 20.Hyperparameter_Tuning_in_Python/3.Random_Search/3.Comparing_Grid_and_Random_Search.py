##                  Comparing Random & Grid Search                  ##
# Which of the following is an advantage of random search?
# Answer: It is more computationally efficient than Grid Search.
# Random search tests a larger space of values so is more likely to get close to the best score, given the same computational resources as Grid Search.





##                  Grid and Random Search Side by Side                  ##
# Part 1
# Sample grid coordinates
grid_combinations_chosen = combinations_list[0:300]

# Print result
print(grid_combinations_chosen)

# output: [[0.01, 5], [0.01, 6], [0.01, 7], [0.01, 8], [0.01, 9], [0.01, 10], [0.01, 11], [0.01, 12], [0.01, 13], [0.01, 14], [0.01, 15], [0.01, 16], [0.01, 17], [0.01, 18], [0.01, 19], [0.01, 20], [0.01, 21], [0.01, 22], [0.01, 23], [0.01, 24], [0.025025125628140705, 5], [0.025025125628140705, 6], [0.025025125628140705, 7], [0.025025125628140705, 8], [0.025025125628140705, 9], [0.025025125628140705, 10], [0.025025125628140705, 11], [0.025025125628140705, 12], [0.025025125628140705, 13], [0.025025125628140705, 14], [0.025025125628140705, 15], [0.025025125628140705, 16], [0.025025125628140705, 17], [0.025025125628140705, 18], [0.025025125628140705, 19], [0.025025125628140705, 20], [0.025025125628140705, 21], [0.025025125628140705, 22], [0.025025125628140705, 23], [0.025025125628140705, 24], [0.04005025125628141, 5], [0.04005025125628141, 6], [0.04005025125628141, 7], [0.04005025125628141, 8], [0.04005025125628141, 9], [0.04005025125628141, 10], [0.04005025125628141, 11], [0.04005025125628141, 12], [0.04005025125628141, 13], [0.04005025125628141, 14], [0.04005025125628141, 15], [0.04005025125628141, 16], [0.04005025125628141, 17], [0.04005025125628141, 18], [0.04005025125628141, 19], [0.04005025125628141, 20], [0.04005025125628141, 21], [0.04005025125628141, 22], [0.04005025125628141, 23], [0.04005025125628141, 24], [0.055075376884422114, 5], [0.055075376884422114, 6], [0.055075376884422114, 7], [0.055075376884422114, 8], [0.055075376884422114, 9], [0.055075376884422114, 10], [0.055075376884422114, 11], [0.055075376884422114, 12], [0.055075376884422114, 13], [0.055075376884422114, 14], [0.055075376884422114, 15], [0.055075376884422114, 16], [0.055075376884422114, 17], [0.055075376884422114, 18], [0.055075376884422114, 19], [0.055075376884422114, 20], [0.055075376884422114, 21], [0.055075376884422114, 22], [0.055075376884422114, 23], [0.055075376884422114, 24], [0.07010050251256282, 5], [0.07010050251256282, 6], [0.07010050251256282, 7], [0.07010050251256282, 8], [0.07010050251256282, 9], [0.07010050251256282, 10], [0.07010050251256282, 11], [0.07010050251256282, 12], [0.07010050251256282, 13], [0.07010050251256282, 14], [0.07010050251256282, 15], [0.07010050251256282, 16], [0.07010050251256282, 17], [0.07010050251256282, 18], [0.07010050251256282, 19], [0.07010050251256282, 20], [0.07010050251256282, 21], [0.07010050251256282, 22], [0.07010050251256282, 23], [0.07010050251256282, 24], [0.08512562814070351, 5], [0.08512562814070351, 6], [0.08512562814070351, 7], [0.08512562814070351, 8], [0.08512562814070351, 9], [0.08512562814070351, 10], [0.08512562814070351, 11], [0.08512562814070351, 12], [0.08512562814070351, 13], [0.08512562814070351, 14], [0.08512562814070351, 15], [0.08512562814070351, 16], [0.08512562814070351, 17], [0.08512562814070351, 18], [0.08512562814070351, 19], [0.08512562814070351, 20], [0.08512562814070351, 21], [0.08512562814070351, 22], [0.08512562814070351, 23], [0.08512562814070351, 24], [0.10015075376884422, 5], [0.10015075376884422, 6], [0.10015075376884422, 7], [0.10015075376884422, 8], [0.10015075376884422, 9], [0.10015075376884422, 10], [0.10015075376884422, 11], [0.10015075376884422, 12], [0.10015075376884422, 13], [0.10015075376884422, 14], [0.10015075376884422, 15], [0.10015075376884422, 16], [0.10015075376884422, 17], [0.10015075376884422, 18], [0.10015075376884422, 19], [0.10015075376884422, 20], [0.10015075376884422, 21], [0.10015075376884422, 22], [0.10015075376884422, 23], [0.10015075376884422, 24], [0.11517587939698493, 5], [0.11517587939698493, 6], [0.11517587939698493, 7], [0.11517587939698493, 8], [0.11517587939698493, 9], [0.11517587939698493, 10], [0.11517587939698493, 11], [0.11517587939698493, 12], [0.11517587939698493, 13], [0.11517587939698493, 14], [0.11517587939698493, 15], [0.11517587939698493, 16], [0.11517587939698493, 17], [0.11517587939698493, 18], [0.11517587939698493, 19], [0.11517587939698493, 20], [0.11517587939698493, 21], [0.11517587939698493, 22], [0.11517587939698493, 23], [0.11517587939698493, 24], [0.13020100502512563, 5], [0.13020100502512563, 6], [0.13020100502512563, 7], [0.13020100502512563, 8], [0.13020100502512563, 9], [0.13020100502512563, 10], [0.13020100502512563, 11], [0.13020100502512563, 12], [0.13020100502512563, 13], [0.13020100502512563, 14], [0.13020100502512563, 15], [0.13020100502512563, 16], [0.13020100502512563, 17], [0.13020100502512563, 18], [0.13020100502512563, 19], [0.13020100502512563, 20], [0.13020100502512563, 21], [0.13020100502512563, 22], [0.13020100502512563, 23], [0.13020100502512563, 24], [0.14522613065326634, 5], [0.14522613065326634, 6], [0.14522613065326634, 7], [0.14522613065326634, 8], [0.14522613065326634, 9], [0.14522613065326634, 10], [0.14522613065326634, 11], [0.14522613065326634, 12], [0.14522613065326634, 13], [0.14522613065326634, 14], [0.14522613065326634, 15], [0.14522613065326634, 16], [0.14522613065326634, 17], [0.14522613065326634, 18], [0.14522613065326634, 19], [0.14522613065326634, 20], [0.14522613065326634, 21], [0.14522613065326634, 22], [0.14522613065326634, 23], [0.14522613065326634, 24], [0.16025125628140705, 5], [0.16025125628140705, 6], [0.16025125628140705, 7], [0.16025125628140705, 8], [0.16025125628140705, 9], [0.16025125628140705, 10], [0.16025125628140705, 11], [0.16025125628140705, 12], [0.16025125628140705, 13], [0.16025125628140705, 14], [0.16025125628140705, 15], [0.16025125628140705, 16], [0.16025125628140705, 17], [0.16025125628140705, 18], [0.16025125628140705, 19], [0.16025125628140705, 20], [0.16025125628140705, 21], [0.16025125628140705, 22], [0.16025125628140705, 23], [0.16025125628140705, 24], [0.17527638190954775, 5], [0.17527638190954775, 6], [0.17527638190954775, 7], [0.17527638190954775, 8], [0.17527638190954775, 9], [0.17527638190954775, 10], [0.17527638190954775, 11], [0.17527638190954775, 12], [0.17527638190954775, 13], [0.17527638190954775, 14], [0.17527638190954775, 15], [0.17527638190954775, 16], [0.17527638190954775, 17], [0.17527638190954775, 18], [0.17527638190954775, 19], [0.17527638190954775, 20], [0.17527638190954775, 21], [0.17527638190954775, 22], [0.17527638190954775, 23], [0.17527638190954775, 24], [0.19030150753768846, 5], [0.19030150753768846, 6], [0.19030150753768846, 7], [0.19030150753768846, 8], [0.19030150753768846, 9], [0.19030150753768846, 10], [0.19030150753768846, 11], [0.19030150753768846, 12], [0.19030150753768846, 13], [0.19030150753768846, 14], [0.19030150753768846, 15], [0.19030150753768846, 16], [0.19030150753768846, 17], [0.19030150753768846, 18], [0.19030150753768846, 19], [0.19030150753768846, 20], [0.19030150753768846, 21], [0.19030150753768846, 22], [0.19030150753768846, 23], [0.19030150753768846, 24], [0.20532663316582916, 5], [0.20532663316582916, 6], [0.20532663316582916, 7], [0.20532663316582916, 8], [0.20532663316582916, 9], [0.20532663316582916, 10], [0.20532663316582916, 11], [0.20532663316582916, 12], [0.20532663316582916, 13], [0.20532663316582916, 14], [0.20532663316582916, 15], [0.20532663316582916, 16], [0.20532663316582916, 17], [0.20532663316582916, 18], [0.20532663316582916, 19], [0.20532663316582916, 20], [0.20532663316582916, 21], [0.20532663316582916, 22], [0.20532663316582916, 23], [0.20532663316582916, 24], [0.22035175879396987, 5], [0.22035175879396987, 6], [0.22035175879396987, 7], [0.22035175879396987, 8], [0.22035175879396987, 9], [0.22035175879396987, 10], [0.22035175879396987, 11], [0.22035175879396987, 12], [0.22035175879396987, 13], [0.22035175879396987, 14], [0.22035175879396987, 15], [0.22035175879396987, 16], [0.22035175879396987, 17], [0.22035175879396987, 18], [0.22035175879396987, 19], [0.22035175879396987, 20], [0.22035175879396987, 21], [0.22035175879396987, 22], [0.22035175879396987, 23], [0.22035175879396987, 24]]


# Part 2
# Create a list of sample indexes
sample_indexes = list(range(0,len(combinations_list)))

# Randomly sample 300 indexes
random_indexes = np.random.choice(sample_indexes, 300, replace=False)


# Part 3
# Use indexes to create random sample
random_combinations_chosen = [combinations_list[index] for index in random_indexes]


# Part 4
# Call the function to produce the visualization
visualize_search(grid_combinations_chosen, random_combinations_chosen)
