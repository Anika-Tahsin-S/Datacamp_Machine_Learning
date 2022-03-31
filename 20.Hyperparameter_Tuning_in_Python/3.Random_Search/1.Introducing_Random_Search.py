# Creating a random sample of hyperparameters
learn_rate_list = np.linspace(0.001, 2, 150)
min_samples_list = list(range(1, 51))

from itertools import product
combinations_list = [list(x) for x in product(learn_rate_list, min_samples_list)]

# Sample hyperparameter combinations for a randomsearch
random_combinations_index = np.random.choice(range(0, len(combinations_list)), 
                                             100, replace = False)
combinations_random_chosen = [combinations_list[x] for x in random_combinations_index]








# --------------------------------------------------------------------------------------------------------- #
##                  Randomly Sample Hyperparameters                  ##
import numpy as np
from itertools import product

# Create a list of values for the learning_rate hyperparameter
learn_rate_list = list(np.linspace(0.01, 1.5, 200))

# Create a list of values for the min_samples_leaf hyperparameter
min_samples_list = list(range(10, 41))

# Combination list
combinations_list = [list(x) for x in product(learn_rate_list, min_samples_list)]

# Sample hyperparameter combinations for a random search.
random_combinations_index = np.random.choice(range(0, len(combinations_list)), 250, replace = False)
combinations_random_chosen = [combinations_list[x] for x in random_combinations_index]

# Print the result
print(combinations_random_chosen)

# output: [[1.305326633165829, 14], [0.6015075376884422, 24], [0.6089949748743718, 35], [1.3128140703517588, 33], [0.3244723618090452, 23], [0.07738693467336683, 27], [0.2421105527638191, 18], [1.1855276381909547, 24], [0.17472361809045225, 11], [0.7662311557788944, 12], [1.0657286432160804, 39], [0.7287939698492463, 22], [1.2604020100502513, 40], [1.0881909547738693, 12], [1.2079899497487436, 39], [1.4475879396984925, 15], [1.327788944723618, 16], [0.7587437185929649, 37], [0.46673366834170854, 10], [0.2720603015075377, 22], [1.2079899497487436, 12], [1.095678391959799, 27], [0.5790452261306532, 31], [0.18969849246231157, 22], [0.17472361809045225, 40], [1.2604020100502513, 39], [0.33195979899497485, 39], [0.05492462311557789, 32], [0.6314572864321608, 40], [0.9758793969849247, 14], [0.7886934673366834, 40], [1.200502512562814, 15], [0.4592462311557789, 22], [0.8635678391959799, 16], [0.7512562814070352, 37], [0.039949748743718594, 18], [0.6838693467336683, 38], [1.1480904522613065, 29], [0.42180904522613066, 31], [0.9010050251256281, 31], [0.26457286432160804, 29], [0.945929648241206, 12], [0.46673366834170854, 16], [1.1630653266331659, 18], [0.7587437185929649, 13], [0.6464321608040201, 16], [0.7812060301507537, 14], [1.0282914572864321, 35], [0.6838693467336683, 12], [0.19718592964824122, 22], [0.12231155778894472, 10], [1.4850251256281406, 15], [0.6988442211055276, 31], [0.4966834170854271, 40], [0.2945226130653266, 31], [1.222964824120603, 28], [1.0582412060301507, 25], [0.7362814070351759, 30], [1.1930150753768844, 17], [0.945929648241206, 21], [1.095678391959799, 35], [0.414321608040201, 39], [1.1555778894472362, 34], [0.7812060301507537, 24], [1.3352763819095477, 29], [0.9309547738693467, 10], [1.0657286432160804, 20], [1.07321608040201, 39], [1.3128140703517588, 21], [1.222964824120603, 18], [0.09984924623115578, 16], [0.36939698492462314, 35], [1.1555778894472362, 33], [0.17472361809045225, 24], [0.024974874371859294, 28], [1.178040201005025, 24], [0.6314572864321608, 29], [0.8261306532663316, 24], [1.4101507537688442, 30], [0.7737185929648241, 29], [1.3951758793969848, 15], [1.222964824120603, 15], [0.7362814070351759, 17], [1.2379396984924622, 24], [0.38437185929648243, 19], [1.07321608040201, 11], [1.4326130653266331, 31], [0.7437688442211056, 17], [1.2154773869346733, 14], [0.6015075376884422, 12], [0.5565829145728644, 19], [1.4401005025125628, 20], [1.222964824120603, 11], [1.2828643216080402, 20], [1.3352763819095477, 18], [1.200502512562814, 24], [1.29035175879397, 16], [0.9010050251256281, 36], [0.9010050251256281, 26], [1.4176381909547737, 17], [1.4326130653266331, 30], [0.3993467336683417, 13], [0.12979899497487438, 20], [1.0881909547738693, 17], [0.024974874371859294, 27], [0.3244723618090452, 10], [0.48170854271356783, 20], [0.6164824120603015, 22], [0.9384422110552764, 28], [0.9833668341708542, 36], [1.1406030150753768, 22], [1.2604020100502513, 20], [0.15974874371859296, 15], [1.2379396984924622, 29], [1.2154773869346733, 28], [0.34693467336683415, 34], [0.6539195979899497, 28], [0.06241206030150754, 26], [1.4925125628140703, 31], [1.4700502512562814, 19], [0.2720603015075377, 33], [0.40683417085427137, 39], [0.2945226130653266, 15], [0.8336180904522613, 13], [0.7886934673366834, 12], [1.3128140703517588, 28], [1.1630653266331659, 22], [0.9084924623115578, 21], [1.3577386934673366, 20], [0.6314572864321608, 15], [0.2421105527638191, 13], [1.2379396984924622, 10], [1.07321608040201, 13], [1.4401005025125628, 31], [0.3544221105527638, 27], [0.2945226130653266, 18], [0.7362814070351759, 13], [0.04743718592964824, 22], [0.8935175879396985, 16], [1.1181407035175879, 31], [1.200502512562814, 31], [1.3352763819095477, 31], [0.8860301507537688, 40], [0.12231155778894472, 17], [0.12231155778894472, 14], [1.1406030150753768, 20], [0.3993467336683417, 21], [0.6164824120603015, 25], [0.7063316582914573, 17], [0.03246231155778895, 10], [0.01, 33], [1.1181407035175879, 11], [0.7587437185929649, 15], [1.1630653266331659, 12], [0.7587437185929649, 23], [0.691356783919598, 35], [1.095678391959799, 25], [0.2421105527638191, 26], [1.0357788944723618, 23], [0.26457286432160804, 12], [1.2753768844221105, 34], [0.6239698492462311, 27], [0.414321608040201, 35], [1.4775376884422111, 32], [0.09984924623115578, 39], [0.5116582914572865, 36], [0.6015075376884422, 14], [0.36939698492462314, 31], [0.3394472361809045, 39], [1.5, 33], [0.06989949748743718, 25], [0.5116582914572865, 10], [1.0058291457286432, 15], [0.5640703517587939, 15], [0.09984924623115578, 31], [1.4176381909547737, 27], [0.6239698492462311, 40], [1.1630653266331659, 29], [1.2379396984924622, 23], [1.2304522613065327, 18], [1.327788944723618, 13], [0.6539195979899497, 32], [0.3094974874371859, 14], [1.178040201005025, 11], [0.7138190954773869, 28], [1.2978391959798994, 37], [0.09236180904522612, 20], [1.178040201005025, 30], [0.4592462311557789, 38], [1.4925125628140703, 23], [1.3951758793969848, 30], [0.3993467336683417, 14], [1.1555778894472362, 36], [1.0807035175879396, 14], [0.13728643216080402, 27], [0.6015075376884422, 11], [0.414321608040201, 31], [1.2304522613065327, 31], [1.4026633165829145, 37], [1.1555778894472362, 39], [0.287035175879397, 21], [1.1555778894472362, 20], [0.8485929648241206, 29], [0.2570854271356784, 19], [1.5, 39], [0.8111557788944723, 29], [1.07321608040201, 19], [0.6164824120603015, 21], [0.8036683417085427, 14], [0.15226130653266332, 20], [0.1822110552763819, 39], [0.5565829145728644, 38], [0.2121608040201005, 38], [0.09236180904522612, 15], [0.7587437185929649, 19], [1.013316582914573, 11], [1.3876884422110551, 22], [0.11482412060301507, 13], [1.3951758793969848, 22], [0.017487437185929648, 16], [0.06241206030150754, 23], [1.4026633165829145, 34], [1.3352763819095477, 23], [1.1705527638190953, 33], [0.2570854271356784, 26], [0.45175879396984925, 25], [0.5191457286432161, 20], [0.5416080402010051, 37], [0.42180904522613066, 37], [0.6838693467336683, 29], [0.7213065326633166, 10], [0.7437688442211056, 37], [0.04743718592964824, 11], [0.05492462311557789, 22], [0.01, 24], [1.3876884422110551, 34], [1.4925125628140703, 24], [0.36939698492462314, 20], [0.18969849246231157, 30], [1.4850251256281406, 35], [1.2753768844221105, 10], [1.2604020100502513, 28], [0.945929648241206, 39], [0.7287939698492463, 30], [0.8785427135678392, 13], [1.4101507537688442, 25], [0.7662311557788944, 14], [0.5116582914572865, 31], [1.1555778894472362, 27], [0.30201005025125627, 21]]






##                  Randomly Sample Hyperparameters                  ##
# Create lists for criterion and max_features
criterion_list = ['gini', 'entropy']
max_feature_list = ["auto", "sqrt", "log2", None]

# Create a list of values for the max_depth hyperparameter
max_depth_list = list(range(3, 56))

# Combination list
combinations_list = [list(x) for x in product(criterion_list, max_feature_list, max_depth_list)]

# Sample hyperparameter combinations for a random search
combinations_random_chosen = random.sample(combinations_list, 150)

# Print the result
print(combinations_random_chosen)

# output: [['entropy', 'log2', 49], ['gini', None, 50], ['entropy', None, 22], ['entropy', 'sqrt', 38], ['gini', None, 17], ['entropy', 'log2', 52], ['gini', 'sqrt', 46], ['entropy', 'log2', 44], ['gini', 'auto', 30], ['entropy', 'sqrt', 45], ['entropy', 'log2', 54], ['gini', 'auto', 54], ['gini', 'auto', 31], ['entropy', None, 40], ['gini', 'log2', 36], ['gini', 'auto', 10], ['gini', None, 26], ['entropy', 'log2', 32], ['entropy', 'auto', 52], ['gini', None, 7], ['entropy', 'log2', 3], ['entropy', 'auto', 15], ['entropy', 'sqrt', 42], ['gini', None, 31], ['gini', 'sqrt', 10], ['gini', 'auto', 44], ['entropy', 'sqrt', 24], ['gini', 'log2', 35], ['entropy', 'sqrt', 50], ['entropy', 'log2', 43], ['entropy', 'log2', 45], ['entropy', 'sqrt', 47], ['entropy', 'log2', 38], ['gini', 'log2', 33], ['gini', 'auto', 51], ['entropy', None, 14], ['entropy', 'auto', 39], ['entropy', 'auto', 47], ['entropy', 'log2', 50], ['gini', 'sqrt', 50], ['gini', 'log2', 43], ['entropy', None, 5], ['gini', None, 18], ['gini', None, 40], ['entropy', 'sqrt', 33], ['gini', 'log2', 17], ['gini', 'sqrt', 53], ['entropy', 'log2', 22], ['entropy', 'auto', 23], ['gini', None, 15], ['gini', 'sqrt', 8], ['gini', None, 4], ['gini', None, 9], ['gini', None, 23], ['gini', 'auto', 26], ['gini', 'auto', 4], ['entropy', 'auto', 20], ['entropy', 'auto', 42], ['gini', 'sqrt', 18], ['entropy', 'log2', 36], ['entropy', 'auto', 27], ['entropy', 'sqrt', 49], ['gini', 'auto', 20], ['gini', 'auto', 22], ['gini', None, 14], ['entropy', 'sqrt', 30], ['gini', 'sqrt', 15], ['entropy', 'log2', 10], ['gini', 'auto', 45], ['gini', None, 52], ['gini', None, 16], ['entropy', None, 41], ['entropy', None, 36], ['entropy', 'auto', 34], ['entropy', 'sqrt', 34], ['gini', 'log2', 10], ['gini', 'auto', 41], ['entropy', 'auto', 36], ['entropy', 'log2', 34], ['entropy', 'auto', 38], ['entropy', 'auto', 37], ['gini', 'log2', 48], ['entropy', 'auto', 30], ['entropy', None, 35], ['entropy', 'auto', 12], ['gini', 'auto', 46], ['entropy', 'sqrt', 52], ['gini', 'sqrt', 37], ['gini', 'auto', 50], ['gini', 'sqrt', 32], ['gini', 'sqrt', 16], ['entropy', 'auto', 24], ['entropy', None, 51], ['entropy', None, 3], ['entropy', 'log2', 33], ['entropy', 'log2', 29], ['gini', 'sqrt', 19], ['gini', 'auto', 6], ['entropy', 'log2', 48], ['entropy', 'auto', 7], ['entropy', None, 10], ['gini', None, 13], ['gini', 'sqrt', 47], ['entropy', 'auto', 8], ['gini', 'sqrt', 7], ['entropy', None, 52], ['gini', 'auto', 42], ['gini', 'sqrt', 27], ['entropy', None, 29], ['entropy', 'log2', 7], ['gini', 'auto', 17], ['entropy', 'sqrt', 48], ['entropy', 'sqrt', 31], ['gini', 'log2', 18], ['entropy', 'log2', 24], ['entropy', 'auto', 10], ['gini', 'auto', 38], ['gini', 'sqrt', 33], ['entropy', 'auto', 25], ['entropy', 'auto', 49], ['gini', None, 5], ['gini', None, 10], ['entropy', None, 24], ['gini', 'auto', 49], ['entropy', 'log2', 20], ['entropy', None, 6], ['gini', None, 37], ['entropy', 'log2', 26], ['gini', 'log2', 9], ['entropy', 'log2', 8], ['entropy', 'auto', 14], ['entropy', None, 37], ['gini', 'auto', 23], ['gini', 'sqrt', 48], ['entropy', None, 39], ['gini', None, 3], ['gini', 'log2', 21], ['entropy', None, 28], ['entropy', 'sqrt', 27], ['gini', 'log2', 14], ['entropy', None, 15], ['entropy', 'log2', 23], ['gini', 'sqrt', 34], ['gini', None, 27], ['gini', 'log2', 8], ['entropy', 'sqrt', 44], ['entropy', 'sqrt', 36], ['gini', 'log2', 23], ['gini', 'log2', 3], ['entropy', 'auto', 45]]






##                  Visualizing a Random Search                  ##
import matplotlib.pyplot as plt

def sample_and_visualize_hyperparameters(n_samples):
    # If asking for all combinations, just return the entire list.
    if n_samples == len(combinations_list):
        combinations_random_chosen = combinations_list
    else:
        combinations_random_chosen = []
        random_combinations_index = np.random.choice(range(0, len(combinations_list)), n_samples, replace=False)
        combinations_random_chosen = [combinations_list[x] for x in random_combinations_index]
    
    # Pull out the X and Y to plot
    rand_y, rand_x = [x[0] for x in combinations_random_chosen], [x[1] for x in combinations_random_chosen]

    # Plot 
    plt.clf() 
    plt.scatter(rand_y, rand_x, c=['blue']*len(combinations_random_chosen))
    plt.gca().set(xlabel='learn_rate', ylabel='min_samples_leaf', title='Random Search Hyperparameters')
    plt.gca().set_xlim([0.01, 1.5])
    plt.gca().set_ylim([10, 29])

# Confirm how many hyperparameter combinations & print
number_combs = len(combinations_list)
print(number_combs)

# Sample and visualise specified combinations
for x in [50, 500, 1500]:
    sample_and_visualize_hyperparameters(x)
    
# Sample all the hyperparameter combinations & visualise
sample_and_visualize_hyperparameters(number_combs)

# output: 2000
