# Tokenization : create a bog of words
# Lowercasing Words
# Lemmatization/Stemming : Shorten words to their root stems
# Removing stop words like 'and', 'the' or punctuations, or unwanted tokens

from collections import Counter
from ntlk.corpus import stopwords
text = """blahblha khaeu hakljs """

tokens = [w for w in word_tokenize(text.lower()) if w.isalpha()]
no_stops = [t for t in tokens if t not in stopwords.wors('english')]
Counter(no_stop).most_common(2)


# ------------------------------------------------------------------------- #
##                   Text Preprocessing Steps                  ##
# Which of the following are useful text preprocessing steps?
# Answer: Lemmatization, lowercasing, removing unwanted tokens.



##                   Text Preprocessing Practice                  ##
# Import WordNetLemmatizer
from nltk.stem import WordNetLemmatizer

# Retain alphabetic words: alpha_only
alpha_only = [t for t in lower_tokens if t.isalpha()]

# Remove all stop words: no_stops
no_stops = [t for t in alpha_only if t not in english_stops]

# Instantiate the WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

# Lemmatize all tokens into a new list: lemmatized
lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops]

# Create the bag-of-words: bow
bow = Counter(lemmatized)

# Print the 10 most common tokens
print(bow.most_common(10))
# output: [('debugging', 40), ('system', 25), ('software', 16), ('bug', 16), ('problem', 15), ('tool', 15), ('computer', 14), ('process', 13), ('term', 13), ('used', 12)]