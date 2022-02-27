# Using N-grams
tv_bi_gram_vec = TfidfVectorizer(ngram_range = (2, 2))

# Fit and apply bigram vectorizer
tv_bi_gram = tv_bi_gram_vec.fit_transform(speech_df['text']

# Print the bigram features
print(tv_bi_gram_vec.get_feature_names())


# Finding common words
tv_df = pd.DataFrame(tv_bi_gram.toarray(), 
                     columns = tv_bi_gram_vec.get_feature_names()).add_prefix('Counts_')
tv_sums = tv.df.sum()
print(tv_sums.head())
print(tv_sums.sort_values(ascending = False).head())



# ------------------------------------------------------------- #
##                   Using Longer n-grams                  ##
# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Instantiate a trigram vectorizer
cv_trigram_vec = CountVectorizer(max_features = 100, stop_words = 'english', ngram_range = (3, 3))

# Fit and apply trigram vectorizer
cv_trigram = cv_trigram_vec.fit_transform(speech_df['text_clean'])

# Print the trigram features
print(cv_trigram_vec.get_feature_names())
# output: ['ability preserve protect', 'agriculture commerce manufactures', 'america ideal freedom', 'amity mutual concession', 'anchor peace home', 'ask bow heads', 
# 'best ability preserve', 'best interests country', 'bless god bless', 'bless united states', 'chief justice mr', 'children children children', 'citizens united states',
#  'civil religious liberty', 'civil service reform', 'commerce united states', 'confidence fellow citizens', 'congress extraordinary session', 
# 'constitution does expressly', 'constitution united states', 'coordinate branches government', 'day task people', 'defend constitution united', 
# 'distinction powers granted', 'distinguished guests fellow', 'does expressly say', 'equal exact justice', 'era good feeling', 'executive branch government', 
# 'faithfully execute office', 'fellow citizens assembled', 'fellow citizens called', 'fellow citizens large', 'fellow citizens world', 'form perfect union', 
# 'general welfare secure', 'god bless america', 'god bless god', 'good greatest number', 'government peace war', 'government united states', 
# 'granted federal government', 'great body people', 'great political parties', 'greatest good greatest', 'guests fellow citizens', 'invasion wars powers', 
# 'land new promise', 'laws faithfully executed', 'letter spirit constitution', 'liberty pursuit happiness', 'life liberty pursuit', 'local self government', 
# 'make hard choices', 'men women children', 'mr chief justice', 'mr majority leader', 'mr president vice', 'mr speaker mr', 'mr vice president', 'nation like person', 
# 'new breeze blowing', 'new states admitted', 'north south east', 'oath prescribed constitution', 'office president united', 'passed generation generation', 
# 'peace shall strive', 'people united states', 'physical moral political', 'policy united states', 'power general government', 'preservation general government', 
# 'preservation sacred liberty', 'preserve protect defend', 'president united states', 'president vice president', 'promote general welfare', 'proof confidence fellow', 
# 'protect defend constitution', 'protection great interests', 'reform civil service', 'reserved states people', 'respect individual human', 'right self government', 
# 'secure blessings liberty', 'south east west', 'sovereignty general government', 'states admitted union', 'territories united states', 'thank god bless', 
# 'turning away old', 'united states america', 'united states best', 'united states government', 'united states great', 'united states maintain', 
# 'united states territory', 'vice president mr', 'welfare secure blessings']






##                   Finding the Most Common Words                  ##
# Create a DataFrame of the features
cv_tri_df = pd.DataFrame(cv_trigram.toarray(), 
                 columns = cv_trigram_vec.get_feature_names()).add_prefix('Counts_')

# Print the top 5 words in the sorted output
print(cv_tri_df.sum().sort_values(ascending = False).head())
# output:
#     Counts_constitution united states    20
#     Counts_people united states          13
#     Counts_mr chief justice              10
#     Counts_preserve protect defend       10
#     Counts_president united states        8
#     dtype: int64