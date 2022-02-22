import pandas as pd
import numpy as np

##                   Exploring Text Vectors, Part 1                  ##
vocab = {v:k for k,v in tfidf_vec.vocabulary_.items()} # swapping key index

# Add in the rest of the parameters
def return_weights(vocab, original_vocab, vector, vector_index, top_n):
    zipped = dict(zip(vector[vector_index].indices, vector[vector_index].data))
    
    # Let's transform that zipped dict into a series
    zipped_series = pd.Series({vocab[i]:zipped[i] for i in vector[vector_index].indices})
    
    # Let's sort the series to pull out the top n weighted words
    zipped_index = zipped_series.sort_values(ascending = False)[:top_n].index
    return [original_vocab[i] for i in zipped_index]

# Print out the weighted words
print(return_weights(vocab, tfidf_vec.vocabulary_, text_tfidf, 8, 3))
# output: [189, 942, 466]

# Collected
text_tfidf[8].indices
text_tfidf[8].toarray()
text_tfidf[8].toarray().shape
text_tfidf[8].data
# ----------------------





##                   Exploring Text Vectors, Part 2                  ##
def words_to_filter(vocab, original_vocab, vector, top_n):
    filter_list = []
    for i in range(0, vector.shape[0]):
    
        # Here we'll call the function from the previous exercise, and extend the list we're creating
        filtered = return_weights(vocab, original_vocab, vector, i, top_n)
        filter_list.extend(filtered)
    # Return the list in a set, so we don't get duplicate word indices
    return set(filter_list)

# Call the function to get the list of word indices
filtered_words = words_to_filter(vocab, tfidf_vec.vocabulary_, text_tfidf, 3)

# By converting filtered_words back to a list, we can use it to filter the columns in the text vector
filtered_text = text_tfidf[:, list(filtered_words)]





##                   Training Naive Bayes With Feature Selection                  ##
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

# Split the dataset according to the class distribution of category_desc, using the filtered_text vector
train_X, test_X, train_y, test_y = train_test_split(filtered_text.toarray(), y, stratify = y)

# Fit the model to the training data
nb.fit(train_X, train_y)

# Print out the model's accuracy
print(nb.score(test_X, test_y))
# output: 0.567741935483871

## The accuracy score wasn't that different from the score at the end of chapter 3. 
# That's okay; the title field is a very small text field, appropriate for demonstrating how filtering vectors works.