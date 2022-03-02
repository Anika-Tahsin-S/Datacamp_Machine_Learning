# sent_tokenize : tokenize a document into sentences
# regexp_tokenize = a string or document based on a regular expression pattern
# TweetTokenizer: special case just for tokenization, allowing to separate hashtags, mentions and lots of exclamation points.


##                   Word Tokenization with NLTK                  ##
# Import necessary modules
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

# Split scene_one into sentences: sentences
sentences = sent_tokenize(scene_one)

# Use word_tokenize to tokenize the fourth sentence: tokenized_sent
tokenized_sent = word_tokenize(sentences[3])

# Make a set of unique tokens in the entire scene: unique_tokens
unique_tokens = set(word_tokenize(scene_one))

# Print the unique tokens result
print(unique_tokens)


# output: {'breadth', 'Ridden', 'European', 'ask', 'there', 'coconuts', ':', 'strangers', 'second', 'court', 'interested', 'Are', 'are', 'since', 'lord', "'ve", 'in', 
# 'covered', 'my', 'coconut', "'m", 'right', 'horse', 'SOLDIER', 'Arthur', 'England', 'grips', '!', 'temperate', 'migrate', 'empty', '1', 'SCENE', 'castle', 'this', 
# 'Listen', 'them', '?', 'master', 'minute', 'its', 'get', 'that', 'creeper', 'zone', 'Court', 'ratios', 'bangin', 'kingdom', '--', 'from', 'agree', 'Supposing', 
# 'Pendragon', 'does', 'Wait', 'have', 'join', 'Pull', 'clop', 'Whoa', 'yet', ']', 'The', 'mean', 'climes', 'two', 'times', 'use', 'held', 'plover', 'pound', 'Oh', 'Yes',
# 'No', 'line', 'these', 'at', 'snows', "'em", 'of', 'Who', 'dorsal', 'What', 'knights', 'Will', 'Halt', 'with', 'Well', 'grip', ',', 'length', 'matter', 'defeator', 
# 'under', 'why', 'or', 'weight', 'carry', 'air-speed', 'order', '2', 'simple', 'through', 'swallow', 'Camelot', 'bird', 'ARTHUR', 'ridden', 'but', '#', 'am', 'halves', 
# 'search', 'other', 'son', "'re", 'house', "'d", 'they', 'King', 'maybe', 'carried', 'A', 'land', 'Saxons', 'must', 'got', 'beat', 'to', 'go', 'Please', 'if', 'Am', 
# 'Uther', 'You', 'needs', 'Found', "'s", 'So', 'who', '.', 'trusty', "n't", 'a', 'Britons', 'In', 'question', 'I', 'your', 'all', 'where', 'goes', 'he', 'martin', 
# 'tell', 'yeah', 'That', 'guiding', 'just', 'then', 'Mercea', 'back', 'Patsy', 'and', 'could', 'bring', 'here', 'wants', 'every', 'servant', 'velocity', 'may', 'winter', 
# 'maintain', 'wind', '[', 'KING', 'speak', 'sun', 'together', 'found', 'anyway', 'It', 'be', 'not', 'is', 'ounce', 'swallows', 'Not', 'wings', 'husk', 'south', 
# 'forty-three', 'seek', 'it', 'five', 'by', 'on', 'using', 'sovereign', 'do', 'Where', 'warmer', 'They', 'strand', 'suggesting', "'", 'fly', 'will', 'you', 'the', 
# 'course', 'feathers', 'an', 'point', 'We', '...', 'non-migratory', 'African', 'carrying', 'tropical', 'But', 'me', 'our', 'one'}






##                   More regex with re.search()                  ##
# Part 1
# Search for the first occurrence of "coconuts" in scene_one: match
match = re.search("coconuts", scene_one)

# Print the start and end indexes of match
print(match.start(), match.end())
# output: 580 588


# Part 2
# Write a regular expression to search for anything in square brackets: pattern1
pattern1 = r"\[.*\]"

# Use re.search to find the first text in square brackets
print(re.search(pattern1, scene_one))
# output: <_sre.SRE_Match object; span=(9, 32), match='[wind] [clop clop clop]'>


# Part 3
# Find the script notation at the beginning of the fourth sentence and print it
pattern2 = r"[\w\s]+:"
print(re.match(pattern2, sentences[3]))
# output: <_sre.SRE_Match object; span=(0, 7), match='ARTHUR:'>