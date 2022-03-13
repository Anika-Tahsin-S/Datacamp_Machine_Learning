# Bag of Words method
from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer()
bow_matrix = vec.fit_transform(corpus)
print(bow_matrix.toarray())




## --------------------------------------------------------------------------------------------------------- ##

##                   Word vectors with a given vocabulary                  ##
# You have been given a corpus of documents and you have computed the vocabulary of the corpus to be the following: V: a, an, and, but, can, come, evening, forever, go, i, men, may, on, the, women
# Which of the following corresponds to the bag of words vector for the document "men may come and men may go but i go on forever"?
# (0, 0, 1, 1, 0, 1, 0, 1, 2, 1, 2, 2, 1, 0, 0)
# Each value in the vector corresponds to the frequency of the corresponding word in the vocabulary.




##                   BoW Model For Movie Taglines                  ##

# corpus is pre given
# The first five taglines in corpus have been printed to the console for you to examine.

# 1            Roll the dice and unleash the excitement!
# 2    Still Yelling. Still Fighting. Still Ready for...
# 3    Friends are the people who let you be yourself...
# 4    Just When His World Is Back To Normal... He's ...
# 5                             A Los Angeles Crime Saga
# Name: tagline, dtype: object

# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Create CountVectorizer object
vectorizer = CountVectorizer()

# Generate matrix of word vectors
bow_matrix = vectorizer.fit_transform(corpus)

# Print the shape of bow_matrix
print(bow_matrix.shape)
# output: (7033, 6614)






##                   Analyzing Dimensionality and Preprocessing                  ##
# The first five lemmatized taglines in lem_corpus have been printed to the console for you to examine. 
# 0    roll dice unleash excitement
# 1           yell fight ready love
# 2    friend people let let forget
# 3      world normal surprise life
# 4          los angeles crime saga
# Name: 1, dtype: object


# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Create CountVectorizer object
vectorizer = CountVectorizer()

# Generate matrix of word vectors
bow_lem_matrix = vectorizer.fit_transform(lem_corpus)

# Print the shape of bow_lem_matrix
print(bow_lem_matrix.shape)

# output: (6959, 5223)





##                   Mapping Feature Indices with Feature Names                  ##
# The sentences are available in a list named corpus and has already been printed to the console.
# ['The lion is the king of the jungle', 'Lions have lifespans of a decade', 'The lion is an endangered species']


# Create CountVectorizer object
vectorizer = CountVectorizer()

# Generate matrix of word vectors
bow_matrix = vectorizer.fit_transform(corpus)

# Convert bow_matrix into a DataFrame
bow_df = pd.DataFrame(bow_matrix.toarray())

# Map the column names to vocabulary 
bow_df.columns = vectorizer.get_feature_names()

# Print bow_df
print(bow_df)
# output:
#        an  decade  endangered  have  is  ...  lion  lions  of  species  the
#     0   0       0           0     0   1  ...     1      0   1        0    3
#     1   0       1           0     1   0  ...     0      1   1        0    0
#     2   1       0           1     0   1  ...     1      0   0        1    1
#     
#     [3 rows x 13 columns]