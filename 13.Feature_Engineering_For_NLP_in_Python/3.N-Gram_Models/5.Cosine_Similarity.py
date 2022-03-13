# Since the cosine score is simply the cosine of the angle between two vectors, its value is bounded between -1 and 1. 
# However, in NLP, document vectors almost always use non-negative weights. 
# Therefore, cosine scores vary between 0 and 1 where 0 indicates no similarity and 1 indicates that the documents are identical.

# Cosine Similarity
from sklearn.metrics.pairwise import cosine_similarity




# --------------------------------------------------------------------------------------------------------- #
##                   Range of Cosine Scores                  ##
# Which of the following is a possible cosine score for a pair of document vectors?
# Answer: 0.86



##                   Computing Dot Product                  ##
import numpy as np
# Initialize numpy vectors
A = np.array([1, 3])
B = np.array([-2, 2])

# Compute dot product
dot_prod = np.dot(A, B)

# Print dot product
print(dot_prod)
# output: 4



##                   Cosine Similarity Matrix of a Corpus                  ##
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

corpus = ['The sun is the largest celestial body in the solar system', 'The solar system consists of the sun and eight revolving planets', 'Ra was the Egyptian Sun God', 'The Pyramids were the pinnacle of Egyptian architecture', 'The quick brown fox jumps over the lazy dog']

# Initialize an instance of tf-idf Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Generate the tf-idf vectors for the corpus
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

# Compute and print the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print(cosine_sim)

# output:
#     [[1.         0.36413198 0.18314713 0.18435251 0.16336438]
#      [0.36413198 1.         0.15054075 0.21704584 0.11203887]
#      [0.18314713 0.15054075 1.         0.21318602 0.07763512]
#      [0.18435251 0.21704584 0.21318602 1.         0.12960089]
#      [0.16336438 0.11203887 0.07763512 0.12960089 1.        ]]