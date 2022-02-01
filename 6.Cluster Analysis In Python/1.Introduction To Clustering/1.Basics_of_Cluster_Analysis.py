 ##                   Pokémon Sightings                  ##
# Import plotting class from matplotlib library
from matplotlib import pyplot as plt

# Create a scatter plot
plt.scatter(x, y)

# Display the scatter plot
plt.show()



 ##                   Pokémon Sightings: Hierarchical Clustering                  ##
# Import linkage and fcluster functions
from scipy.cluster.hierarchy import linkage, fcluster
from matplotlib import pyplot as plt
from seaborn as sns, pandas as pd

x_coordinates = [80.1, 93.1, 86.6, 98.5, 86.4, 9.5, 15.2, 3.4, 10.4, 20.3, 
                    44.2, 56.8, 49.2, 62.5, 44.0]
y_coordinates = [87.2, 96.1, 95.6, 92.4, 92.4, 57.7, 49.4, 47.3, 59.1, 
                    55.5, 25.6, 2.1, 10.9, 24.1, 10.3]
df = pd.DataFrame({'x_coord':x_coordinates, 'y_coord':y_coordinates})
#df.head()


# Use the linkage() function to compute distance
Z = linkage(df, method = "ward")

# Generate cluster labels for each data point with two clusters
df['cluster_labels'] = fcluster(Z, 2, criterion = "maxclust")
# Plot the points with seaborn
# 'x' and 'y' are columns of X and Y coordinates of the locations of sightings, stored in a pandas DataFrame, df.
sns.scatterplot(x = "x", y = "y", hue = "cluster_labels", data = df)
#sns.scatterplot(data = df, x = "x_coord", y = "y_coord", hue = "cluster_labels", palette=  "RdGy")
plt.show()



 ##                   Pokémon Sightings: KMeans Clustering                  ##
# Import kmeans and vq functions
from scipy.cluster.vq import kmeans, vq
from matplotlib import pyplot as plt
from seaborn as sns, pandas as pd

import random
random.seed(1000, 2000)

x_coordinates = [80.1, 93.1, 86.6, 98.5, 86.4, 9.5, 15.2, 3.4, 10.4, 20.3, 
                    44.2, 56.8, 49.2, 62.5, 44.0]
y_coordinates = [87.2, 96.1, 95.6, 92.4, 92.4, 57.7, 49.4, 47.3, 59.1, 
                    55.5, 25.6, 2.1, 10.9, 24.1, 10.3]


#df.dtypes
#df = df.apply(lambda x: x.astype("float"))

# Compute cluster centers
centroids,_ = kmeans(df, 2)

# Assign cluster labels
df['cluster_labels'], _ = vq(df, centroids)

# Plot the points with seaborn
# 'x' and 'y' are columns of X and Y coordinates of the locations of sightings, stored in a pandas DataFrame, df.
sns.scatterplot(x = "x", y = "y", hue = "cluster_labels", data = df)
#sns.scatterplot(data = df, x = "x_coord", y = "y_coord", hue = "cluster_labels", palette=  "RdGy")
plt.show()