##                   Correlation Intuition                  ##

# What statement on correlations is correct?
# The correlation coefficient of A to B is equal to that of B to A.
# This is why you can drop half of the correlation matrix without losing information.


##                   Inspecting the Correlation Matrix                  ##

# What is the correlation coefficient between wrist and ankle circumference?
# 0.702178. Quite a strong, positive correlation.



##                   Visualizing The Correlation Matrix                  ##
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = ansur_df[['headbreadth', 'headcircumference', 'headlength', 'tragiontopofhead']]
# Create the correlation matrix
corr = df.corr()


# Full plot
cmap = sns.diverging_palette(h_neg = 10, h_pos = 240, as_camp = True)
# Draw the heatmap
sns.heatmap(corr, center = 0, cmap = cmap, linewidths = 1, annot = True, fmt = '.2f')


# Focused plot
# Generate a mask for the upper triangle 
mask = np.triu(np.ones_like(corr, dtype = bool))

# Add the mask to the heatmap
sns.heatmap(corr, cmap = 'coolwarm', mask = mask, center = 0, linewidths = 1, annot = True, fmt = ".2f")
plt.show()


# Which two features have the strongest correlation?
# The buttock and crotch height have a 0.93 correlation coefficient.