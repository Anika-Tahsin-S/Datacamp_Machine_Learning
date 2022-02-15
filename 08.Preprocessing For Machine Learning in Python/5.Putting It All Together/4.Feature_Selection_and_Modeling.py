##                   Selecting the Ideal Dataset                  ##
vocab = {v:k for k,v in vec.vocabulary_.items()}

# Check the correlation between the seconds, seconds_log, and minutes columns
print(ufo[['seconds', 'seconds_log', 'minutes']].corr())

# Make a list of features to drop
to_drop = ["city", "country", "date", "desc", "lat", "length_of_time", "long", "minutes", "recorded", "seconds", "state"]

# Drop those features
ufo_dropped = ufo.drop(to_drop, axis = 1)

# Let's also filter some words out of the text vector we created
filtered_words = words_to_filter(vocab, vec.vocabulary_, desc_tfidf, 4)





##                   Modeling the UFO Dataset, Part 1                  ##
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier()

# Collected
ufo_dropped.columns
X = ufo_dropped[['seconds_log', 'changing', 'chevron', 'cigar', 'circle', 'cone',
       'cross', 'cylinder', 'diamond', 'disk', 'egg', 'fireball', 'flash',
       'formation', 'light', 'other', 'oval', 'rectangle', 'sphere',
       'teardrop', 'triangle', 'unknown', 'month', 'year']]

y = ufo_dropped['country_enc']
X.shape, y.shape
# -----------------------------

# Take a look at the features in the X set of data
print(X.columns)

# Split the X and y sets using train_test_split, setting stratify=y
train_X, test_X, train_y, test_y = train_test_split(X, y, stratify = y)

# Fit knn to the training sets
knn.fit(train_X, train_y)

# Print the score of knn on the test sets
print(knn.score(test_X, test_y))
# output:
#    Index(['seconds_log', 'changing', 'chevron', 'cigar', 'circle', 'cone',
#           'cross', 'cylinder', 'diamond', 'disk', 'egg', 'fireball', 'flash',
#           'formation', 'light', 'other', 'oval', 'rectangle', 'sphere',
#           'teardrop', 'triangle', 'unknown', 'month', 'year'],
#          dtype='object')
#    0.8693790149892934





##                   Modeling the UFO Dataset, Part 2                  ##
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()

# Use the list of filtered words we created to filter the text vector
filtered_text = desc_tfidf[:, list(filtered_words)]

# Split the X and y sets using train_test_split, setting stratify=y 
train_X, test_X, train_y, test_y = train_test_split(filtered_text.toarray(), y, stratify = y)

# Fit nb to the training sets
nb.fit(train_X, train_y)

# Print the score of nb on the test sets
nb.score(test_X, test_y)
# output: 0.16274089935760172