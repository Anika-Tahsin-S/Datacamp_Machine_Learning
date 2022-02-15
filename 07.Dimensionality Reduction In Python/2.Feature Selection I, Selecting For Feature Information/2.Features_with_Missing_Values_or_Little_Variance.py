##                   Finding a Good Variance Threshold                  ##
# Pre-loaded head measurements
head_df = ansur_df[['headbreadth', 'headcircumference', 'headlength', 'tragiontopofhead']]
head_df['n_hairs'] = 10000 

# Part 1
# Create the boxplot
head_df.boxplot()

plt.show()

# Part 2
# Normalize the data
normalized_df = head_df / head_df.mean()

normalized_df.boxplot()
plt.show()

# Part 3
# Normalize the data
normalized_df = head_df / head_df.mean()

# Print the variances of the normalized data
print(normalized_df.var())

# Part 4
# Inspect the printed variances. If you want to remove the 2 very low variance features.
# What would be a good variance threshold?
# A threshold of 1.0e-03 (0.001) will remove the two low variance features.



##                   Features With Low Variance                  ##
from sklearn.feature_selection import VarianceThreshold

# Create a VarianceThreshold feature selector
sel = VarianceThreshold(threshold = 0.001)

# Fit the selector to normalized head_df
sel.fit(head_df / head_df.mean())

# Create a boolean mask
mask = sel.get_support()

# Apply the mask to create a reduced dataframe
reduced_df = head_df.loc[:, mask]

print("Dimensionality reduced from {} to {}.".format(head_df.shape[1], reduced_df.shape[1]))



##                   Removing Features With many missing Values                  ##
import pandas as pd

# Part 1
school_df = pd.read_csv('Public_Schools2.csv')

# In what range lies highest ratio of missing values for a single feature in the dataset?
school_df.isna().sum() / len(school_df)
# Between 0.9 and 1.0.

# Part 2
# Create a boolean mask on whether each feature less than 50% missing values.
mask = school_df.isna().sum() / len(school_df) < 0.5

# Create a reduced dataset by applying the mask
reduced_df = school_df.loc[:, mask]

print(school_df.shape)
print(reduced_df.shape)