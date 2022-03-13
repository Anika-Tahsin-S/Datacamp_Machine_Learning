# Generating cosine similarity matrix
from sklearn.metrics.pairwise import linear_kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matix)



# --------------------------------------------------------------------------------------------------------- #
##                   Comparing linear_kernel and cosine_similarity                  ##
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel

# Part 1
# Record start time
start = time.time()

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Print cosine similarity matrix
print(cosine_sim)

# Print time taken
print("Time taken: %s seconds" %(time.time() - start))
# output:
    [[1.         0.         0.         ... 0.         0.         0.        ]
     [0.         1.         0.         ... 0.         0.         0.        ]
     [0.         0.         1.         ... 0.         0.01418221 0.        ]
     ...
     [0.         0.         0.         ... 1.         0.01589009 0.        ]
     [0.         0.         0.01418221 ... 0.01589009 1.         0.        ]
     [0.         0.         0.         ... 0.         0.         1.        ]]
    Time taken: 0.4001162052154541 seconds


# Part 2
# Record start time
start = time.time()

# Compute cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Print cosine similarity matrix
print(cosine_sim)

# Print time taken
print("Time taken: %s seconds" %(time.time() - start))
# output:
    [[1.         0.         0.         ... 0.         0.         0.        ]
     [0.         1.         0.         ... 0.         0.         0.        ]
     [0.         0.         1.         ... 0.         0.01418221 0.        ]
     ...
     [0.         0.         0.         ... 1.         0.01589009 0.        ]
     [0.         0.         0.01418221 ... 0.01589009 1.         0.        ]
     [0.         0.         0.         ... 0.         0.         1.        ]]
    Time taken: 0.35402488708496094 seconds




##                   Plot Recommendation Engine                  ##
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the TfidfVectorizer 
tfidf = TfidfVectorizer(stop_words = 'english')

# Construct the TF-IDF matrix
tfidf_matrix = tfidf.fit_transform(movie_plots)

# Generate the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
 
# Generate recommendations 
print(get_recommendations('The Dark Knight Rises', cosine_sim, indices))

# output:
#     1                              Batman Forever
#     2                                      Batman
#     3                              Batman Returns
#     8                  Batman: Under the Red Hood
#     9                            Batman: Year One
#     10    Batman: The Dark Knight Returns, Part 1
#     11    Batman: The Dark Knight Returns, Part 2
#     5                Batman: Mask of the Phantasm
#     7                               Batman Begins
#     4                              Batman & Robin
#     Name: title, dtype: object



##                   The Recommender Function                  ##
# Given a dataset metadata that consists of the movie titles and overviews.

# Import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Create TfidfVectorizer object
vectorizer = TfidfVectorizer()

# Generate matrix of word vectors
tfidf_matrix = vectorizer.fit_transform(vectorizer)

# Print the shape of tfidf_matrix
print(tfidf_matrix.shape)

#                title                                            tagline
# 938  Cinema Paradiso  A celebration of youth, friendship, and the ev...
# 630         Spy Hard  All the action. All the women. Half the intell...
# 682        Stonewall                    The fight for the right to love
# 514           Killer                    You only hurt the one you love.
# 365    Jason's Lyric                                   Love is courage.

# Generate mapping between titles and index
indices = pd.Series(metadata.index, index = metadata['title']).drop_duplicates()

def get_recommendations(title, cosine_sim, indices):
    # Get index of movie that matches title
    idx = indices[title]
    # Sort the movies based on the similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)
    # Get the scores for 10 most similar movies
    sim_scores = sim_scores[1:11]
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    # Return the top 10 most similar movies
    return metadata['title'].iloc[movie_indices]





##                   TED Talk Recommender                  ##
# Initialize the TfidfVectorizer 
tfidf = TfidfVectorizer(stop_words = 'english')

# Construct the TF-IDF matrix
tfidf_matrix = tfidf.fit_transform(transcripts)

# Generate the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix)
 
# Generate recommendations 
print(get_recommendations('5 ways to kill your dreams', cosine_sim, indices))
# output:
#     453             Success is a continuous journey
#     157                        Why we do what we do
#     494                   How to find work you love
#     149          My journey into movies that matter
#     447                        One Laptop per Child
#     230             How to get your ideas to spread
#     497         Plug into your hard-wired happiness
#     495    Why you will fail to have a great career
#     179             Be suspicious of simple stories
#     53                          To upgrade is human
#     Name: title, dtype: object