from scipy.cluster.heirarchy import linkage
import pandas as pd
import random, timeit

points = 100
df = pd.DataFrame({'x' : random.sample(range(0, points), points),
                    'y' : random.sample(range(0, points), points)})
%timeit linkage(df[['x', 'y']], method = 'ward', metric = 'euclidean')


##                                Exercise                               ##
##                   Timing Run of Hierarchical Clustering                  ##

%timeit linkage(comic_con[['x_scaled', 'y_scaled']], method = 'ward', metric = 'euclidean')



##                   FIFA 18: Exploring Defenders                  ##
fifa[['sliding_tackle', 'aggression']].head()
fifa['scaled_sliding_tackle'] = whiten(fifa.sliding_tackle)
fifa['scaled_aggression'] = whiten(fifa.aggression)

# Fit the data into a hierarchical clustering algorithm
distance_matrix = linkage(fifa[['scaled_sliding_tackle', 'scaled_aggression']], 'ward')

# Assign cluster labels to each row of data
fifa['cluster_labels'] = fcluster(distance_matrix, 3, criterion='maxclust')

# Display cluster centers of each cluster
print(fifa[['scaled_sliding_tackle', 'scaled_aggression', 'cluster_labels']].groupby('cluster_labels').mean())

# Create a scatter plot through seaborn
sns.scatterplot(x = 'scaled_sliding_tackle', y = 'scaled_aggression', hue = 'cluster_labels', data = fifa)
plt.show()