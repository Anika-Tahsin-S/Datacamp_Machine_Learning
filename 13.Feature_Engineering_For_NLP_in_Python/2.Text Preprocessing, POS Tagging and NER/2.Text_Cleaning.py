# Removing stopwords using spacy
stopwords = spacy.lang.en.stop_words.STOP_WORDS

a_lemmas = [lemma for lemma in a_lemmas 
            if lemma.isalpha() and lemma not in stopwords]
print(' '.join(a_lemmas))




## --------------------------------------------------------------------------------------------------------- ##

##                   Cleaning a blog post                  ##
blog = '\nTwenty-first-century politics has witnessed an alarming rise of populism in the U.S. and Europe. The first warning signs came with the UK Brexit Referendum vote in 2016 swinging in the way of Leave. This was followed by a stupendous victory by billionaire Donald Trump to become the 45th President of the United States in November 2016. Since then, Europe has seen a steady rise in populist and far-right parties that have capitalized on Europe’s Immigration Crisis to raise nationalist and anti-Europe sentiments. Some instances include Alternative for Germany (AfD) winning 12.6% of all seats and entering the Bundestag, thus upsetting Germany’s political order for the first time since the Second World War, the success of the Five Star Movement in Italy and the surge in popularity of neo-nazism and neo-fascism in countries such as Hungary, Czech Republic, Poland and Austria.\n'
stopwords = ['fifteen', 'noone', 'whereupon', 'could', 'ten', 'all', 'please', 'indeed', 'whole', 'beside', 'therein', 'using', 'but', 'very', 'already', 'about', 'no', 'regarding', 'afterwards', 'front', 'go', 'in', 'make', 'three', 'here', 'what', 'without', 'yourselves', 'which', 'nothing', 'am', 'between', 'along', 'herein', 'sometimes', 'did', 'as', 'within', 'elsewhere', 'was', 'forty', 'becoming', 'how', 'will', 'other', 'bottom', 'these', 'amount', 'across', 'the', 'than', 'first', 'namely', 'may', 'none', 'anyway', 'again', 'eleven', 'his', 'meanwhile', 'name', 're', 'from', 'some', 'thru', 'upon', 'whither', 'he', 'such', 'down', 'my', 'often', 'whether', 'made', 'while', 'empty', 'two', 'latter', 'whatever', 'cannot', 'less', 'many', 'you', 'ours', 'done', 'thus', 'since', 'everything', 'for', 'more', 'unless', 'former', 'anyone', 'per', 'seeming', 'hereafter', 'on', 'yours', 'always', 'due', 'last', 'alone', 'one', 'something', 'twenty', 'until', 'latterly', 'seems', 'were', 'where', 'eight', 'ourselves', 'further', 'themselves', 'therefore', 'they', 'whenever', 'after', 'among', 'when', 'at', 'through', 'put', 'thereby', 'then', 'should', 'formerly', 'third', 'who', 'this', 'neither', 'others', 'twelve', 'also', 'else', 'seemed', 'has', 'ever', 'someone', 'its', 'that', 'does', 'sixty', 'why', 'do', 'whereas', 'are', 'either', 'hereupon', 'rather', 'because', 'might', 'those', 'via', 'hence', 'itself', 'show', 'perhaps', 'various', 'during', 'otherwise', 'thereafter', 'yourself', 'become', 'now', 'same', 'enough', 'been', 'take', 'their', 'seem', 'there', 'next', 'above', 'mostly', 'once', 'a', 'top', 'almost', 'six', 'every', 'nobody', 'any', 'say', 'each', 'them', 'must', 'she', 'throughout', 'whence', 'hundred', 'not', 'however', 'together', 'several', 'myself', 'i', 'anything', 'somehow', 'or', 'used', 'keep', 'much', 'thereupon', 'ca', 'just', 'behind', 'can', 'becomes', 'me', 'had', 'only', 'back', 'four', 'somewhere', 'if', 'by', 'whereafter', 'everywhere', 'beforehand', 'well', 'doing', 'everyone', 'nor', 'five', 'wherein', 'so', 'amongst', 'though', 'still', 'move', 'except', 'see', 'us', 'your', 'against', 'although', 'is', 'became', 'call', 'have', 'most', 'wherever', 'few', 'out', 'whom', 'yet', 'be', 'own', 'off', 'quite', 'with', 'and', 'side', 'whoever', 'would', 'both', 'fifty', 'before', 'full', 'get', 'sometime', 'beyond', 'part', 'least', 'besides', 'around', 'even', 'whose', 'hereby', 'up', 'being', 'we', 'an', 'him', 'below', 'moreover', 'really', 'it', 'of', 'our', 'nowhere', 'whereby', 'too', 'her', 'toward', 'anyhow', 'give', 'never', 'another', 'anywhere', 'mine', 'herself', 'over', 'himself', 'to', 'onto', 'into', 'thence', 'towards', 'hers', 'nevertheless', 'serious', 'under', 'nine']

import spacy
# Load model and create Doc object
nlp = spacy.load('en_core_web_sm')
doc = nlp(blog)

# Generate lemmatized tokens
lemmas = [token.lemma_ for token in doc]

# Remove stopwords and non-alphabetic tokens
a_lemmas = [lemma for lemma in lemmas 
            if lemma.isalpha() and lemma not in stopwords]

# Print string after text cleaning
print(' '.join(a_lemmas))
# output:
#     century politic witness alarming rise populism europe warning sign come uk brexit referendum vote swinging way leave follow stupendous victory billionaire donald trump president united states november europe steady rise populist far right party capitalize europe immigration crisis raise nationalist anti europe sentiment instance include alternative germany afd win seat enter bundestag upset germany political order time second world war success star movement italy surge popularity neo nazism neo fascism country hungary czech republic poland austria





##                   Cleaning TED Talks in a Dataframe                  ##
# Function to preprocess text
def preprocess(text):
  	# Create Doc object
    doc = nlp(text, disable = ['ner', 'parser'])
    # Generate lemmas
    lemmas = [token.lemma_ for token in doc]
    # Remove stopwords and non-alphabetic characters
    a_lemmas = [lemma for lemma in lemmas 
            if lemma.isalpha() and lemma not in stopwords]
    
    return ' '.join(a_lemmas)
  
# Apply preprocess to ted['transcript']
ted['transcript'] = ted['transcript'].apply(preprocess)
print(ted['transcript'])
# output:
#     0     talk new lecture ted illusion create ted try r...
#     1     representation brain brain break left half log...
#     2     great honor today share digital universe creat...
#     3     passion music technology thing combination thi...
#     4     use want computer new program programming requ...
#     5     neuroscientist mixed background physics medici...
#     6     pat mitchell day january begin like work love ...
#     7     taylor wilson year old nuclear physicist littl...
#     8     grow northern ireland right north end absolute...
#     9     publish article new york times modern love col...
#     10    joseph member parliament kenya picture maasai ...
#     11    hi talk little bit music machine life specific...
#     12    hi let ask audience question lie child raise h...
#     13    historical record allow know ancient greeks dr...
#     14    good morning little boy experience change life...
#     15    slide year ago time short slide morning time w...
#     16    like world like share year old love story poor...
#     17    fail woman fail feminist passionate opinion ge...
#     18    revolution century significant longevity revol...
#     19    today baffle lady observe shell soul dwellsand...
#     Name: transcript, dtype: object