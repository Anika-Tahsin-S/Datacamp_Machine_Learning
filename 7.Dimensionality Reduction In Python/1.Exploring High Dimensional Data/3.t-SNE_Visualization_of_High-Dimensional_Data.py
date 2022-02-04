##                   t-SNE Intuition                  ##
# A good use case to use t-SNE is when you want to visually explore the patterns in a high dimensional dataset.



##                   Fitting t-SNE To The ANSUR Data                  ##
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Non-numerical columns in the dataset
non_numeric = ['Branch', 'Gender', 'Component']

# Drop the non-numerical columns from df
df_numeric = df.drop(non_numeric, axis = 1)

# Create a t-SNE model with learning rate 50
m = TSNE(learning_rate = 50)

# Fit and transform the t-SNE model on the numeric dataset
tsne_features = m.fit_transform(df_numeric)
print(tsne_features.shape)



##                   t-SNE Visualisation of Dimensionality                  ##

tsne = pd.DataFrame(tsne_features, columns = ['x', 'y'])
df = ansur_df.join(tsne)

# Part 1
# Color the points according to Army Component
sns.scatterplot(x = "x", y = "y", hue = 'Component', data = df)

# Show the plot
plt.show()
# Part 2
# Color the points by Army Branch
sns.scatterplot(x = "x", y = "y", hue = 'Branch', data = df)

# Show the plot
plt.show()

# Part 3
# Color the points by Army Branch
sns.scatterplot(x = "x", y = "y", hue = 'Gender', data = df)

# Show the plot
plt.show()