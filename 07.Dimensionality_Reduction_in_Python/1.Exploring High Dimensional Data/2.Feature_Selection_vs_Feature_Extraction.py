##                   Visually Detecting Redundant Features                  ##
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline


# Pre-loaded Data
female = pd.read_csv('ANSUR_II_FEMALE.csv')
male = pd.read_csv('ANSUR_II_MALE.csv') 
ansur_df = pd.concat([female, male])
ansur_df['body_height'] = ansur_df['stature_m']
ansur_df['n_legs'] = 2
ansur_df_1 = ansur_df[['Gender', 'weight_kg', 'stature_m', 'body_height']]
ansur_df_2 = ansur_df[['Gender', 'footlength', 'headlength', 'n_legs']]


# Part 1
# Create a pairplot and color the points using the 'Gender' feature
sns.pairplot(ansur_df_1, hue = 'Gender', diag_kind = 'hist')

# Show the plot
plt.show()

# Part 2
# Remove one of the redundant features
reduced_df = ansur_df_1.drop('stature_m', axis = 1)

# Create a pairplot and color the points using the 'Gender' feature
sns.pairplot(reduced_df, hue = 'Gender')

# Show the plot
plt.show()


# Part 3
# Create a pairplot and color the points using the 'Gender' feature
sns.pairplot(ansur_df_2, hue = 'Gender', diag_kind='hist')

# Show the plot
plt.show()

# Part 4
# Remove the redundant feature
reduced_df = ansur_df_2.drop('n_legs', axis = 1)

# Create a pairplot and color the points using the 'Gender' feature
sns.pairplot(reduced_df, hue = 'Gender', diag_kind = 'hist')

# Show the plot
plt.show()



##                   Advantage of Feature Selection                  ##
# The selected features remain unchanged, and are therefore easy to interpret.
# Extracted features can be quite hard to interpret.