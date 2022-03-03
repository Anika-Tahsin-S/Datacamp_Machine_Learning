##                   What is tf-idf?                  ##
# You want to calculate the tf-idf weight for the word "computer", which appears five times in a document containing 100 words. Given a corpus containing 200 documents, with 20 documents mentioning the word "computer", tf-idf can be calculated by multiplying term frequency with inverse document frequency.
# Term frequency = percentage share of the word compared to all tokens in the document Inverse document frequency = logarithm of the total number of documents in a corpora divided by the number of documents containing the term

# Which of the below options is correct?
import math
(5 / 100) * math.log(200 / 20)

# Answer: (5 / 100) * log(200 / 20)




##                   Tf-idf with Wikipedia                  ##
from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidModel

# Part 1
# Create a new TfidfModel using the corpus: tfidf
tfidf = TfidfModel(corpus)

# Calculate the tfidf weights of doc: tfidf_weights
tfidf_weights = tfidf[doc]

# Print the first five weights
print(tfidf_weights[:5])
# output:
#     [(24, 0.0022836332291091273), (39, 0.0043409401554717324), (41, 0.008681880310943465), (55, 0.011988285029371418), (56, 0.005482756770026296)]


# Part 2
# Sort the weights from highest to lowest: sorted_tfidf_weights
sorted_tfidf_weights = sorted(tfidf_weights, key = lambda w: w[1], reverse = True)

# Print the top 5 weighted words
for term_id, weight in sorted_tfidf_weights[:5]:
    print(dictionary.get(term_id), weight)
# output:
#     reverse 0.4884961428651127
#     infringement 0.18674529210288995
#     engineering 0.16395041814479536
#     interoperability 0.12449686140192663
#     reverse-engineered 0.12449686140192663