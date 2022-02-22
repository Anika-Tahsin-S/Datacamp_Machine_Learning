##                   Finding The Number of Dimensions in a Dataset                  ##
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

pokemon_df = pd.read_csv('pokemon.csv')
pokemon_df.head()
pokemon_df.shape



##                   Removing Features Without Variance                  ##
# Leave this list as is
number_cols = ['HP', 'Attack', 'Defense'] # 'Generation' was removed as all data was 1

# Remove the feature without variance from this list
non_number_cols = ['Name', 'Type'] # 'Legendary' was removed as all data was False

# Create a new dataframe by subselecting the chosen features
df_selected = pokemon_df[number_cols + non_number_cols]

# Prints the first 5 lines of the new dataframe
print(df_selected.head())