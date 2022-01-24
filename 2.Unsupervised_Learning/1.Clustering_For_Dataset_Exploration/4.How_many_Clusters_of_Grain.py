# Given an array samples containing the measurements (such as area, perimeter, length, and several others) of samples of grain. 
# What's a good number of clusters in this case?
# KMeans and PyPlot (plt) have already been imported for you.
# This dataset was sourced from the UCI Machine Learning Repository.

ks = range(1, 6)
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters = k)
    
    # Fit model to samples
    model.fit(samples)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()