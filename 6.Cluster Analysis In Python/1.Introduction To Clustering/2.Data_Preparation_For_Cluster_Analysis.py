 ##                   Normalize Basic List Data                  ##
# Import the whiten function
from scipy.cluster.vq import whiten

goals_for = [4,3,2,3,1,1,2,0,1,4]

# Use the whiten() function to standardize the data
scaled_data = whiten(goals_for)
print(scaled_data)




 ##                   Visualize Normalized Data                  ##
# Matplotlib
from matplotlib import pyplot as plt

# Plot original data
plt.plot(goals_for, label='original')
# Plot scaled data
plt.plot(scaled_data, label='scaled')
# Show the legend in the plot
plt.legend()
# Display the plot
plt.show()

# Seaborn
from seaborn as sns, pandas as pd

_ = sns.lineplot(x = range(len(goals_for)), y = goals_for, label = "original")
_ = sns.lineplot(x = range(len(goals_for)), y = scaled_data, label = 'scaled')
plt.show()




 ##                   Normalized of Small Numbers                  ##
# Prepare data
rate_cuts = [0.0025, 0.001, -0.0005, -0.001, -0.0005, 0.0025, -0.001, -0.0015, -0.001, 0.0005]

# Use the whiten() function to standardize the data
scaled_data = whiten(rate_cuts)

# Plot original data
plt.plot(rate_cuts, label = 'original')

# Plot scaled data
plt.plot(scaled_data, label = 'scaled')

plt.legend()
plt.show()



 ##                   FIFA 18: Normalized Data                  ##
from scipy.cluster.vq import whiten
from matplotlib.pyplot as plt

# The data for this exercise is stored in a pandas DataFrame, fifa.whiten from scipy.cluster.vq and matplotlib.pyplot as plt have been pre-loaded.
fifa = pd.read_csv("datasets/fifa.csv")
fifa.head()

# Scale wage and value
fifa['scaled_wage'] = whiten(fifa['eur_wage'])
fifa['scaled_value'] = whiten(fifa['eur_value'])

# Plot the two columns in a scatter plot
fifa.plot(x='scaled_wage', y='scaled_value', kind = 'scatter')
plt.show()

# Check mean and standard deviation of scaled values
print(fifa[['scaled_wage', 'scaled_value']].describe())