##                   Filtering Out Highly Correlated Features                  ##
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the correlation matrix and take the absolute value
corr_matrix = ansur_df.corr().abs()

# Create a True/False mask and apply it
mask = np.triu(np.ones_like(corr_matrix, dtype = bool))
tri_df = corr_matrix.mask(mask)

# List column names of highly correlated features (r > 0.95)
to_drop = [c for c in tri_df.columns if any(tri_df[c] >  0.95)]

# Drop the features in the to_drop list
reduced_df = ansur_df.drop(to_drop, axis = 1)

print("The reduced dataframe has {} columns.".format(reduced_df.shape[1]))



##                   Nuclear Energy and Pool Drownings                  ##
weird_df = pd.DataFrame([[421.0, 728.3], [465.0, 753.9], [494.0, 768.8], [538.0, 780.1], [430.0, 763.7], [530.0, 788.5], [511.0, 782.0], [600.0, 787.2], [582.0, 806.4], [605.0, 806.2], [603.0, 798.9]], columns = ['pool_drownings', 'nuclear_energy'])

# Put nuclear energy production on the x-axis and the number of pool drownings on the y-axis
sns.scatterplot(x = 'nuclear_energy', y = 'pool_drownings', data = weird_df)

# Print the first five lines of weird_df
print(weird_df.head())

# Print out the correlation matrix of weird_df
print(weird_df.corr())


# What can you conclude from the strong correlation (r=0.9) between these features?
# Not much, correlation does not imply causation.
# While the example is silly, you'll be amazed how often people misunderstand correlation vs causation.