##                   Bag-of-words picker                  ##
#It's time for a quick check on your understanding of bag-of-words. Which of the below options, with basic nltk tokenization, map the bag-of-words for the following text?

# "The cat is in the box. The cat box."

from nltk.tokenize import word_tokenize
from collections import Counter

Counter(word_tokenize('"The cat is in the box. The cat box."'))
Counter.most_common(2)

# Answer: ('The', 2), ('box', 2), ('.', 2), ('cat', 2), ('is', 1), ('in', 1), ('the', 1)





##                   Building a Counter with bag-of-words                  ##
# Import Counter
from collections import Counter

# Tokenize the article: tokens
tokens = word_tokenize(article)

# Convert the tokens into lowercase: lower_tokens
lower_tokens = [t.lower() for t in tokens]

# Create a Counter with the lowercase tokens: bow_simple
bow_simple = Counter(lower_tokens)

# Print the 10 most common tokens
print(bow_simple.most_common(10))
# output: [(',', 151), ('the', 150), ('.', 89), ('of', 81), ("''", 68), ('to', 63), ('a', 60), ('in', 44), ('and', 41), ('debugging', 40)]