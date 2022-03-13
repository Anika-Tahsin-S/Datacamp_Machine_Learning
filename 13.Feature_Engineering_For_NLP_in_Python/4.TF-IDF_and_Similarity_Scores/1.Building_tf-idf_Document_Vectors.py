##                   tf-idf Weight of Commonly Occurring Words                  ##
# The word bottle occurs 5 times in a particular document D and also occurs in every document of the corpus. What is the tf-idf weight of bottle in D?
# Answer: 0
# In fact, the tf-idf weight for bottle in every document will be 0. This is because the inverse document frequency is constant across documents in a corpus and since bottle occurs in every document, its value is log(1), which is 0.



##                   tf-idf Vectors For TED Talks                  ##
# Given a corpus ted which contains the transcripts of 500 TED Talks.
# Import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Create TfidfVectorizer object
vectorizer = TfidfVectorizer()

# Generate matrix of word vectors
tfidf_matrix = vectorizer.fit_transform(ted)

# Print the shape of tfidf_matrix
print(tfidf_matrix.shape)
# output: (500, 29158)