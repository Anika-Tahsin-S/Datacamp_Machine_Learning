##                         Choosing a tokenizer                        ##
# Given the following string, which of the below patterns is the best tokenizer? 
# If possible, you want to retain sentence punctuation as separate tokens, but have '#1' remain a single token.
# The string is available in your workspace as my_string, 
# And the patterns have been pre-loaded as pattern1, pattern2, pattern3, and pattern4, respectively.

Additionally, regexp_tokenize has been imported from nltk.tokenize. You can use regexp_tokenize(string, pattern) with my_string and one of the patterns as arguments to experiment for yourself and see which is the best tokenizer.

from nltk.tokenazie import regexp_tokenize
my_string = "SOLDIER #1: Found them? In Mercea? The coconut's tropical!"
# Search for the first occurrence of "coconuts" in scene_one: match
match = re.search("coconuts", scene_one)

# Print the start and end indexes of match
print(match.start(), match.end())
print(regexp_tokenize(my_string, pattern1))
# ['SOLDIER', '1', 'Found', 'them', '?', 'In', 'Mercea', '?', 'The', 'coconut', 's', 'tropical', '!']

print(regexp_tokenize(my_string, pattern2))
# ['SOLDIER', '#1', 'Found', 'them', '?', 'In', 'Mercea', '?', 'The', 'coconut', 's', 'tropical', '!']

# Answer: r"(\w+|#\d|\?|!)"




##                         Regex with NLTK Tokenization                        ##
# Part 1
# Import the necessary modules
from nltk.tokenize import regexp_tokenize
from nltk.tokenize import TweetTokenizer

# Part 2
# Define a regex pattern to find hashtags: pattern1
pattern1 = r"#\w+"
# Use the pattern on the first tweet in the tweets list
hashtags = regexp_tokenize(tweets[0], pattern1)
print(hashtags)
# output: ['#nlp', '#python']

# Part 3
# Write a pattern that matches both mentions (@) and hashtags
pattern2 = r"([@|#]\w+)"
# Use the pattern on the last tweet in the tweets list
mentions_hashtags = regexp_tokenize(tweets[-1], pattern2)
print(mentions_hashtags)
# output: ['@datacamp', '#nlp', '#python']

# Part 4
# Use the TweetTokenizer to tokenize all tweets into one list
tknzr = TweetTokenizer()
all_tokens = [tknzr.tokenize(t) for t in tweets]
print(all_tokens)
# output: [['This', 'is', 'the', 'best', '#nlp', 'exercise', 'ive', 'found', 'online', '!', '#python'], ['#NLP', 'is', 'super', 'fun', '!', '<3', '#learning'], ['Thanks', '@datacamp', ':)', '#nlp', '#python']]





##                         Non-ascii Tokenization                        ##
from nltk.tokenize import regexp_tokenize, word_tokenize

# Unicode ranges for emoji are:

#('\U0001F300'-'\U0001F5FF'), ('\U0001F600-\U0001F64F'), ('\U0001F680-\U0001F6FF'), and ('\u2600'-\u26FF-\u2700-\u27BF')

german_text = Wann gehen wir Pizza essen? 🍕 Und fährst du mit Über? 🚕

# Tokenize and print all words in german_text
all_words = word_tokenize(german_text)
print(all_words)

# Tokenize and print only capital words
capital_words = r"[A-ZÜ]\w+"
print(regexp_tokenize(german_text, capital_words))

# Tokenize and print only emoji
emoji = "['\U0001F300-\U0001F5FF'|'\U0001F600-\U0001F64F'|'\U0001F680-\U0001F6FF'|'\u2600-\u26FF\u2700-\u27BF']"
print(regexp_tokenize(german_text, emoji))
# output: ['Wann', 'gehen', 'wir', 'Pizza', 'essen', '?', '🍕', 'Und', 'fährst', 'du', 'mit', 'Über', '?', '🚕']
#         ['Wann', 'Pizza', 'Und', 'Über']
#         ['🍕', '🚕']