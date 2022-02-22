import pandas as pd
import numpy as np
import re



##                   Engineering Features From Strings - Extraction                  ##
hiking["Length"].dropna(inplace = True)

# Write a pattern to extract numbers and decimals
def return_mileage(length):
    pattern = re.compile(r"\d+\.\d+")
    
    # Search the text for matches
    mile = re.match(pattern, length)
    
    # If a value is returned, use group(0) to return the found value
    if mile is not None:
        return float(mile.group(0))
        
# Apply the function to the Length column and take a look at both columns
hiking["Length_num"] = hiking["Length"].apply(lambda row: return_mileage(row))
print(hiking[["Length", "Length_num"]].head())




##                   Engineering Features From Strings - tf/idf                  ##
from sklearn.feature_extraction.text import TfidfVectorizer

# Need to drop NaN for train_test_split
volunteer_csv = pd.read_csv('volunteer_opportunities.csv')
volunteer = volunteer_csv[['category_desc','title']]
volunteer.dropna(inplace = True)
# or
# volunteer = volunteer.dropna(subset = ['category_desc'], axis = 0)


# Take the title text
title_text = volunteer["title"]

# Create the vectorizer method
tfidf_vec = TfidfVectorizer()

# Transform the text into tf-idf vectors
text_tfidf = tfidf_vec.fit_transform(title_text)




##                   Text Classification Using tf/idf Vectors                  ##
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

text_tfidf.toarray().shape

# Split the dataset according to the class distribution of category_desc
y = volunteer["category_desc"]

y.shape

X_train, X_test, y_train, y_test = train_test_split(text_tfidf.toarray(), y, stratify = y)

# Fit the model to the training data
nb.fit(X_train, y_train)

# Print out the model's accuracy
print(nb.score(X_test, y_test))

# 0.567741935483871