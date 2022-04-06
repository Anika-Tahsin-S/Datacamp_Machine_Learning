## Finding Missing Data
df.isnull().head(1)
df.isnull().sum()


## Numerical Missing Data
from sklearn.impute import SimpleImputer

mean_imputer = SimpleImputer(strategy = 'mean')
constant_imputer = SimpleImputer(strategy = 'constant', fill_value = -999)

dff[['num']] = mean_imputer.fit_transform(df[['num']])


## Categorical Missing Data
from sklearn.impute import SimpleImputer

frequent_imputer = SimpleImputer(strategy = 'most_frequent')
constant_imputer = SimpleImputer(strategy = 'constant', fill_value = 'MISS')

dff[['cat']] = mean_imputer.fit_transform(df[['cat']])



## ====================================================================================================== ##