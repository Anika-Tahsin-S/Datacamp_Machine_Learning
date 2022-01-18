from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

iris = datasets.load_iris()
type(iris)
iris.keys()

print(type(iris.data))
print(type(iris.target))

iris.data.shape
iris.target_names

X = iris.data
y = iris.target
df = pd.DataFrame(X, columns = iris.feature_names)
pd.plotting.scatter_matrix(df, c = y, figsize = [9, 9], s = 150, marker = 'D')

# Numerical EDA

df.head()
df.info()
df.describe()

# Visual EDA
# Generating countplots for 'satellite'
plt.figure()
sns.countplot(x = 'satellite', hue = 'party', data = df, palette = 'RdBu')
plt.xticks([0,1], ['No', 'Yes'])
plt.show()

# Generating countplots for 'missile'
plt.figure()
sns.countplot(x = 'missile', hue = 'party', data = df, palette = 'RdBu')
plt.xticks([0,1], ['No', 'Yes'])
plt.show()