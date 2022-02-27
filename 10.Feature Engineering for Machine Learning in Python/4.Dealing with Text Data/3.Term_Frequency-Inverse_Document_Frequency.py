# TF-IDF = [(Count of word occurances / Total words in document)] / log(Number of docs word is in / Total number of docs)
# Vectorizer || Putting it all together
train_tv_df = pd.DataFrame(train_tv_transformed.toarray(), columns = tv.get_feature_names()).add_prefix('TFIDF_')

train_speech_df = pd.concat([train_speech_df, train_tv_df], axis = 1, sort = False)

# Inspecting the transforms
examine_row = train_tv_df.iloc[0]
print(examine_row.sort_values(ascending = False))

# Applying the vectorizer to new data
test_tv_transformed = tv.transform(test_df['tect_clean'])

test_tv_df = pd.DataFrame(test_tv_transformed.toarray(), columns = tv.get_feature_names()).add_prefix('TFIDF_')

test_speech_df = pd.concat([test_seech_df, test_tv_df], axis = 1, sort = False)






# ------------------------------------------------------ #
##                   Tf-idf                  ##
# Import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Instantiate TfidfVectorizer
tv = TfidfVectorizer(max_features = 100, stop_words = 'english')

# Fit the vectroizer and transform the data
tv_transformed = tv.fit_transform(speech_df['text_clean'])

# Create a DataFrame with these features
tv_df = pd.DataFrame(tv_transformed.toarray(), 
                     columns = tv.get_feature_names()).add_prefix('TFIDF_')
print(tv_df.head())
# output:
#        TFIDF_action  TFIDF_administration  TFIDF_america  TFIDF_american  TFIDF_americans  ...  TFIDF_war  TFIDF_way  TFIDF_work  TFIDF_world  TFIDF_years
#     0         0.000                 0.133          0.000           0.105              0.0  ...      0.000      0.061       0.000        0.046        0.053
#     1         0.000                 0.261          0.266           0.000              0.0  ...      0.000      0.000       0.000        0.000        0.000
#     2         0.000                 0.092          0.157           0.073              0.0  ...      0.024      0.000       0.000        0.064        0.073
#     3         0.000                 0.093          0.000           0.000              0.0  ...      0.037      0.000       0.039        0.096        0.000
#     4         0.041                 0.040          0.000           0.031              0.0  ...      0.094      0.000       0.000        0.055        0.063
# 
#     [5 rows x 100 columns]





##                   Inspecting Tf-idf Values                  ##
# Isolate the row to be examined
sample_row = tv_df.iloc[0]

# Print the top 5 words of the sorted output
print(sample_row.sort_values(ascending = False).head())
# output:
#     TFIDF_government    0.367
#     TFIDF_public        0.333
#     TFIDF_present       0.315
#     TFIDF_duty          0.239
#     TFIDF_country       0.230
#     Name: 0, dtype: float64





##                   Transforming Unseen Data                  ##
# Instantiate TfidfVectorizer
tv = TfidfVectorizer(max_features = 100, stop_words = 'english')

# Fit the vectroizer and transform the data
tv_transformed = tv.fit_transform(train_speech_df['text_clean'])

# Transform test data
test_tv_transformed = tv.transform(test_speech_df['text_clean'])

# Create new features for the test set
test_tv_df = pd.DataFrame(test_tv_transformed.toarray(), 
                          columns = tv.get_feature_names()).add_prefix('TFIDF_')
print(test_tv_df.head())
# output:
#        TFIDF_action  TFIDF_administration  TFIDF_america  TFIDF_american  TFIDF_authority  ...  TFIDF_war  TFIDF_way  TFIDF_work  TFIDF_world  TFIDF_years
#     0         0.000                 0.030          0.234           0.083            0.000  ...      0.079      0.033       0.000        0.300        0.135
#     1         0.000                 0.000          0.547           0.037            0.000  ...      0.053      0.067       0.079        0.278        0.126
#     2         0.000                 0.000          0.127           0.135            0.000  ...      0.043      0.054       0.096        0.225        0.044
#     3         0.037                 0.067          0.267           0.031            0.040  ...      0.030      0.038       0.236        0.237        0.062
#     4         0.000                 0.000          0.222           0.157            0.028  ...      0.021      0.081       0.120        0.300        0.153
#     
#     [5 rows x 100 columns]