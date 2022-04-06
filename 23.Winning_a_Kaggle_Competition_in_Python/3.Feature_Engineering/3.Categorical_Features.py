## Label encoding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df['cat_encoded'] = le.fit_transform(df['cat'])


## One-Hot encoding
ohe = pd.get_dummies(df['cat'], prefix = 'ohe_cat')

df.drop('cat', axis = 1, inplace = True)

df = pd.concat([df, ohe], axis = 1)


## Binary Features
binary_feature

le = LabelEncoder()
binary_features['binary_encoded'] = le.fit_transform(binary_feature['binary_feat'])





## ====================================================================================================== ##