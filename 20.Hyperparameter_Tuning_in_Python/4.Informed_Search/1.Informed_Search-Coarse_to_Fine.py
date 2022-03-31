import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def visualize_hyperparameter(name):
    plt.clf()
    plt.scatter(results_df[name],results_df['accuracy'], c=['blue']*500)
    plt.gca().set(xlabel='{}'.format(name), ylabel='accuracy', title='Accuracy for different {}s'.format(name))
    plt.gca().set_ylim([0,100])


##                  Visualizing Coarse to Fine                  ##
# Confirm the size of the combinations_list
print(len(combinations_list))

# Sort the results_df by accuracy and print the top 10 rows
print(results_df.sort_values(by = 'accuracy', ascending = False).head(10))

# Confirm which hyperparameters were used in this search
print(results_df.columns)

# Call visualize_hyperparameter() with each hyperparameter in turn
visualize_hyperparameter('max_depth')
visualize_hyperparameter('min_samples_leaf')
visualize_hyperparameter('learn_rate')

# output:
#     10000
#         max_depth  min_samples_leaf  learn_rate  accuracy
#     1          10                14    0.477450        97
#     4           6                12    0.771275        97
#     2           7                14    0.050067        96
#     3           5                12    0.023356        96
#     5          13                11    0.290470        96
#     6           6                10    0.317181        96
#     7          19                10    0.757919        96
#     8           2                16    0.931544        96
#     9          16                13    0.904832        96
#     10         12                13    0.891477        96
#     Index(['max_depth', 'min_samples_leaf', 'learn_rate', 'accuracy'], dtype='object')






##                  Coarse to Fine Iterations                  ##
# Part 1
def visualize_first():
    for name in results_df.columns[0:2]:
        plt.clf()
        plt.scatter(results_df[name],results_df['accuracy'], c=['blue']*500)
        plt.gca().set(xlabel='{}'.format(name), ylabel='accuracy', title='Accuracy for different {}s'.format(name))
        plt.gca().set_ylim([0,100])
        x_line = 20
        if name == "learn_rate":
            x_line = 1
        plt.axvline(x=x_line, color="red", linewidth=4)

# Use the provided function to visualize the first results
visualize_first()

# Part 2
# Use the provided function to visualize the first results
# visualize_first()

# Create some combinations lists & combine
max_depth_list = list(range(1, 21))
learn_rate_list = np.linspace(0.001, 1, 50)

# Part 3
def visualize_second():
    for name in results_df2.columns[0:2]:
        plt.clf()
        plt.scatter(results_df[name],results_df['accuracy'], c=['blue']*500)
        plt.gca().set(xlabel='{}'.format(name), ylabel='accuracy', title='Accuracy for different {}s'.format(name))
        plt.gca().set_ylim([0,100])
        x_line = 20
        if name == "learn_rate":
            x_line = 1
        plt.axvline(x=x_line, color="red", linewidth=4)

# Call the function to visualize the second results
visualize_second()
