import pandas as pd
import numpy as np

hiking = pd.read_json('hiking.json')
hiking.head()



##                   Encoding Categorical Variables - Binary                  ##
# Pandas
hiking["Accesible_enc"] = hiking["Accessible"].apply(lambda val: 1 if val == "y" else 0)

# Scikit-learn
from sklearn.preprocessing import LabelEncoder

# Set up the LabelEncoder object
enc = LabelEncoder()

# Apply the encoding to the "Accessible" column
hiking['Accessible_enc'] = enc.fit_transform(hiking['Accessible'])

# Compare the two columns
print(hiking[['Accessible', 'Accessible_enc']].head())




##                   Encoding Categorical Variables - one-hot                  ##
# Transform the category_desc column
category_enc = pd.get_dummies(volunteer["category_desc"])

# Take a look at the encoded columns
print(category_enc.head())