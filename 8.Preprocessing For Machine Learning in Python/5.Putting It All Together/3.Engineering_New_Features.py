##                   Encoding Categorical Variables                  ##
# Use Pandas to encode us values as 1 and others as 0
ufo["country_enc"] = ufo["country"].apply(lambda row: 1 if row == 'us' else 0)

# Print the number of unique type values
print(len(ufo.type.unique()))

# Create a one-hot encoded set of the type values
type_set = pd.get_dummies(ufo.type)

# Concatenate this set back to the ufo DataFrame
ufo = pd.concat([ufo, type_set], axis = 1)




##                   Features From Dates                  ##
# Look at the first 5 rows of the date column
print(ufo["date"].head())
# print(ufo.date.head())

# Extract the month from the date column
ufo["month"] = ufo["date"].apply(lambda row: row.month)
# ufo["month"] = ufo["date"].dt.month

# Extract the year from the date column
ufo["year"] = ufo["date"].apply(lambda row: row.year)
# ufo["year"] = ufo["date"].dt.year

# Take a look at the head of all three columns
print(ufo[["date", "month", "year"]].head())




##                   Text Vectorization                  ##
# Take a look at the head of the desc field
print(ufo.desc.head())

# Create the tfidf vectorizer object
vec = TfidfVectorizer()

# Use vec's fit_transform method on the desc field
desc_tfidf = vec.fit_transform(ufo["desc"])

# Look at the number of columns this creates
print(desc_tfidf.shape)
# output:
#    0    It was a large&#44 triangular shaped flying ob...
#    1    Dancing lights that would fly around and then ...
#    2    Brilliant orange light or chinese lantern at o...
#    3    Bright red light moving north to north west fr...
#    4    North-east moving south-west. First 7 or so li...
#    Name: desc, dtype: object
#
#    (1866, 3422)