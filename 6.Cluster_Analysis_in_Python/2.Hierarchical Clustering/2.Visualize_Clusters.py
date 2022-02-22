# Matplotlib
from matplotlib import pyplot as plt
import pandas as pd

df = pd.DataFrame({'x' : [2, 3, 5, 6, 2],
                    'y' : [1, 1, 5, 5, 2],
                    'labels' : ['A', 'A', 'B', 'B', 'A']})
colors = {'A' : 'red', 'B' : 'blue'}
df.plot.scatter(x = 'x',
                y = 'y',
                c = df['labels'].apply(lambda x: colors[x]))
plt.show()


# Seaborn
from matplotlib import pyplot as plt
import seaborn as sns

df = pd.DataFrame({'x' : [2, 3, 5, 6, 2],
                    'y' : [1, 1, 5, 5, 2],
                    'labels' : ['A', 'A', 'B', 'B', 'A']})

sns.scatterplot(x = 'x',
                y = 'y',
                hue = 'labels',
                data = df)
plt.show()



##                   Exercises                  ##
# The data is stored in a pandas DataFrame, comic_con. 
comic_con = pd.read_csv("datasets/comic_con.csv")
comic_con.head()

# x_scaled and y_scaled are the column names of the standardized X and Y coordinates of people at a given point in time. 
# cluster_labels has the cluster labels. 
# A linkage object is stored in the variable distance_matrix.

# Import the fcluster and linkage functions
from scipy.cluster.hierarchy import linkage, fcluster
# Use the linkage() function
distance_matrix = linkage(comic_con[['x_scaled', 'y_scaled']], method = "ward", metric = 'euclidean')
# Assign cluster labels
comic_con['cluster_labels'] = fcluster(distance_matrix, 2, criterion = 'maxclust')


##                   Visualize Clusters With Matplotlib                  ##
# Import the pyplot class
from matplotlib import pyplot as plt

# Define a colors dictionary for clusters
colors = {1:'red', 2:'blue'}

# Plot a scatter plot
comic_con.plot.scatter(x = 'x_scaled', 
                	   y = 'y_scaled',
                	   c = comic_con['cluster_labels'].apply(lambda x: colors[x]))
plt.show()


##                   Visualize Clusters With Seaborn                  ##
# Import the seaborn module
import seaborn as sns

# Plot a scatter plot using seaborn
sns.scatterplot(x = 'x_scaled', 
                y = 'y_scaled', 
                hue = 'cluster_labels', 
                data = comic_con)
plt.show()