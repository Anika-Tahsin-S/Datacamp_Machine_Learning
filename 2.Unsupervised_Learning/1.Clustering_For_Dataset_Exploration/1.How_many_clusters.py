# Given an array points of size 300x2, where each row gives the (x, y) co-ordinates of a point on a map. Make a scatter plot of these pointsand use the scatter plot to guess how many clusters there are.
# matplotlib.pyplot has already been imported as plt

xs = points[:, 0]
ys = points[:, 1]
plt.scatter(xs, ys)
plt.show()

#3 Clusters were seen