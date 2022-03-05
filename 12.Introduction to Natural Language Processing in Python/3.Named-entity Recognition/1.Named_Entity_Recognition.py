##                   NER with NLTK                  ##
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
# Tokenize the article into sentences: sentences
sentences = nltk.sent_tokenize(article)

# Tokenize each sentence into words: token_sentences
token_sentences = [nltk.word_tokenize(sent) for sent in sentences]

# Tag each tokenized sentence into parts of speech: pos_sentences
pos_sentences = [nltk.pos_tag(sent) for sent in token_sentences] 

# Create the named entity chunks: chunked_sentences
chunked_sentences = nltk.ne_chunk_sents(pos_sentences, binary = True)

# Test for stems of the tree with 'NE' tags
for sent in chunked_sentences:
    for chunk in sent:
        if hasattr(chunk, "label") and chunk.label() == "NE":
            print(chunk)
# output:
#     (NE Uber/NNP)
#     (NE Beyond/NN)
#     (NE Apple/NNP)
#     (NE Uber/NNP)
#     (NE Uber/NNP)
#     (NE Travis/NNP Kalanick/NNP)
#     (NE Tim/NNP Cook/NNP)
#     (NE Apple/NNP)
#     (NE Silicon/NNP Valley/NNP)
#     (NE CEO/NNP)
#     (NE Yahoo/NNP)
#     (NE Marissa/NNP Mayer/NNP)




##                   Charting Practice                  ##
# Part 1
# Create the defaultdict: ner_categories
ner_categories = defaultdict(int)

# Part 2
# Create the nested for loop
for sent in chunked_sentences:
    for chunk in sent:
        if hasattr(chunk, 'label'):
            ner_categories[chunk.label()] += 1
            
# Create a list from the dictionary keys for the chart labels: labels
labels = list(ner_categories.keys())

# Part 3
# Create a list of the values: values
values = [ner_categories.get(v) for v in labels]

# Create the pie chart
plt.pie(values, labels = labels, autopct = '%1.1f%%', startangle = 140)

# Display the chart
plt.show()





##                   Stanford library with NLTK                  ##
# When using the Stanford library with NLTK, what is needed to get started?
# NLTK, the Stanford Java Libraries and some environment variables to help with integration.