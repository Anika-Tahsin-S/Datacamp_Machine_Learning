# Tokenization using spacy
import spacy
nlp = spacy.load('en_core_web_sm') # en_core_web_sm = language object
string = "OKay! It's fun."
doc = nlp(string)
tokens = [token.text for token in doc]
print(tokens)


# Lemmatization using spacy
import spacy
nlp = spacy.load('en_core_web_sm') # en_core_web_sm = language object
string = "OKay! It's fun."
doc = nlp(string)
lemmas = [token.lemma_ for token in doc]
print(lemmas)

## --------------------------------------------------------------------------------------------------------- ##

##                   Identifying lemmas                  ##
# Identify the list of words from the choices which do not have the same lemma.
# Answer: Car, Bike, Truck, Bus




##                   Tokenizing the Gettysburg Address                  ##
f1 = open("gettysburg.txt"); forbes = f.readline(); f.close()

import spacy

# Load the en_core_web_sm model
nlp = spacy.load('en_core_web_sm')

# Create a Doc object
doc = nlp(gettysburg)

# Generate the tokens
tokens = [token.text for token in doc]
print(tokens)
# output:
#     ['Four', 'score', 'and', 'seven', 'years', 'ago', 'our', 'fathers', 'brought', 'forth', 'on', 'this', 'continent', ',', 'a', 'new', 'nation', ',', 'conceived', 
# 'in', 'Liberty', ',', 'and', 'dedicated', 'to', 'the', 'proposition', 'that', 'all', 'men', 'are', 'created', 'equal', '.', 'Now', 'we', "'re", 'engaged', 'in', 'a', 
# 'great', 'civil', 'war', ',', 'testing', 'whether', 'that', 'nation', ',', 'or', 'any', 'nation', 'so', 'conceived', 'and', 'so', 'dedicated', ',', 'can', 'long', 
# 'endure', '.', 'We', "'re", 'met', 'on', 'a', 'great', 'battlefield', 'of', 'that', 'war', '.', 'We', "'ve", 'come', 'to', 'dedicate', 'a', 'portion', 'of', 'that', 
# 'field', ',', 'as', 'a', 'final', 'resting', 'place', 'for', 'those', 'who', 'here', 'gave', 'their', 'lives', 'that', 'that', 'nation', 'might', 'live', '.', 'It', 
# "'s", 'altogether', 'fitting', 'and', 'proper', 'that', 'we', 'should', 'do', 'this', '.', 'But', ',', 'in', 'a', 'larger', 'sense', ',', 'we', 'ca', "n't", 'dedicate', 
# '-', 'we', 'can', 'not', 'consecrate', '-', 'we', 'can', 'not', 'hallow', '-', 'this', 'ground', '.', 'The', 'brave', 'men', ',', 'living', 'and', 'dead', ',', 'who', 
# 'struggled', 'here', ',', 'have', 'consecrated', 'it', ',', 'far', 'above', 'our', 'poor', 'power', 'to', 'add', 'or', 'detract', '.', 'The', 'world', 'will', 'little', 
# 'note', ',', 'nor', 'long', 'remember', 'what', 'we', 'say', 'here', ',', 'but', 'it', 'can', 'never', 'forget', 'what', 'they', 'did', 'here', '.', 'It', 'is', 'for', 
# 'us', 'the', 'living', ',', 'rather', ',', 'to', 'be', 'dedicated', 'here', 'to', 'the', 'unfinished', 'work', 'which', 'they', 'who', 'fought', 'here', 'have', 'thus', 
# 'far', 'so', 'nobly', 'advanced', '.', 'It', "'s", 'rather', 'for', 'us', 'to', 'be', 'here', 'dedicated', 'to', 'the', 'great', 'task', 'remaining', 'before', 'us', 
# '-', 'that', 'from', 'these', 'honored', 'dead', 'we', 'take', 'increased', 'devotion', 'to', 'that', 'cause', 'for', 'which', 'they', 'gave', 'the', 'last', 'full', 
# 'measure', 'of', 'devotion', '-', 'that', 'we', 'here', 'highly', 'resolve', 'that', 'these', 'dead', 'shall', 'not', 'have', 'died', 'in', 'vain', '-', 'that', 'this', 
# 'nation', ',', 'under', 'God', ',', 'shall', 'have', 'a', 'new', 'birth', 'of', 'freedom', '-', 'and', 'that', 'government', 'of', 'the', 'people', ',', 'by', 'the', 
# 'people', ',', 'for', 'the', 'people', ',', 'shall', 'not', 'perish', 'from', 'the', 'earth', '.']





##                   Lemmatizing the Gettysburg Address                  ##
gettysburg = "Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal. Now we're engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. We're met on a great battlefield of that war. We've come to dedicate a portion of that field, as a final resting place for those who here gave their lives that that nation might live. It's altogether fitting and proper that we should do this. But, in a larger sense, we can't dedicate - we can not consecrate - we can not hallow - this ground. The brave men, living and dead, who struggled here, have consecrated it, far above our poor power to add or detract. The world will little note, nor long remember what we say here, but it can never forget what they did here. It is for us the living, rather, to be dedicated here to the unfinished work which they who fought here have thus far so nobly advanced. It's rather for us to be here dedicated to the great task remaining before us - that from these honored dead we take increased devotion to that cause for which they gave the last full measure of devotion - that we here highly resolve that these dead shall not have died in vain - that this nation, under God, shall have a new birth of freedom - and that government of the people, by the people, for the people, shall not perish from the earth."

# Part 1
# Print the gettysburg address
print(gettysburg)

# Part 2
import spacy

# Load the en_core_web_sm model
nlp = spacy.load('en_core_web_sm')

# Create a Doc object
doc = nlp(gettysburg)

# Generate lemmas
lemmas = [token.lemma_ for token in doc]

# Part 3
# Convert lemmas into a string
print(' '.join(lemmas))
# output:
#     four score and seven year ago -PRON- father bring forth on this continent , a new nation , conceive in liberty , and dedicate to the proposition that all man be create equal . now -PRON- be engage in a great civil war , test whether that nation , or any nation so conceive and so dedicated , can long endure . -PRON- be meet on a great battlefield of that war . -PRON- have come to dedicate a portion of that field , as a final resting place for those who here give -PRON- life that that nation may live . -PRON- be altogether fitting and proper that -PRON- should do this . but , in a large sense , -PRON- can not dedicate - -PRON- can not consecrate - -PRON- can not hallow - this ground . the brave man , living and dead , who struggle here , have consecrate -PRON- , far above -PRON- poor power to add or detract . the world will little note , nor long remember what -PRON- say here , but -PRON- can never forget what -PRON- do here . -PRON- be for -PRON- the living , rather , to be dedicate here to the unfinished work which -PRON- who fight here have thus far so nobly advanced . -PRON- be rather for -PRON- to be here dedicate to the great task remain before -PRON- - that from these honor dead -PRON- take increase devotion to that because for which -PRON- give the last full measure of devotion - that -PRON- here highly resolve that these dead shall not have die in vain - that this nation , under god , shall have a new birth of freedom - and that government of the people , by the people , for the people , shall not perish from the earth .