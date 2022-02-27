##                   Counting Words (I)                  ##
# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Instantiate CountVectorizer
cv = CountVectorizer()

# Fit the vectorizer
cv.fit(speech_df['text_clean'])

# Print feature names
print(cv.get_feature_names())




##                   Counting Words (II)                  ##
# Part 1
# Apply the vectorizer
cv_transformed = cv.transform(speech_df['text_clean'])

# Print the full array
cv_array = cv_transformed.toarray()
print(cv_array)
# output:
#     [[0 0 0 ... 0 0 0]
#      [0 0 0 ... 0 0 0]
#      [0 1 0 ... 0 0 0]
#      ...
#      [0 1 0 ... 0 0 0]
#      [0 0 0 ... 0 0 0]
#      [0 0 0 ... 0 0 0]]


# Part 2
# Apply the vectorizer
cv_transformed = cv.transform(speech_df['text_clean'])

# Print the full array
cv_array = cv_transformed.toarray()

# Print the shape of cv_array
print(cv_array.shape)
# output: (58, 9043)




##                   Limiting your Features                  ##
# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Specify arguements to limit the number of features generated
cv = CountVectorizer(min_df = 0.2, max_df = 0.8)

# Fit, transform, and convert into array
cv_transformed = cv.fit_transform(speech_df['text_clean'])
cv_array = cv_transformed.toarray()

# Print the array shape
print(cv_transformed.shape)
# output: (58, 818)
# The number of features (unique words) greatly reduced from 9043 to 818






##                   Text to DataFrame                  ##
# Create a DataFrame with these features
cv_df = pd.DataFrame(cv_array, 
                     columns = cv.get_feature_names()).add_prefix('Counts_')

# Add the new columns to the original DataFrame
speech_df_new = pd.concat([speech_df, cv_df], axis = 1, sort = False)
print(speech_df_new.head())
# output:
#                     Name         Inaugural Address                      Date                                               text                                         text_clean  ...  Counts_years  \
#     0  George Washington   First Inaugural Address  Thursday, April 30, 1789  Fellow-Citizens of the Senate and of the House...  fellow citizens of the senate and of the house...  ...             1   
#     1  George Washington  Second Inaugural Address     Monday, March 4, 1793  Fellow Citizens:  I AM again called upon by th...  fellow citizens   i am again called upon by th...  ...             0   
#     2         John Adams         Inaugural Address   Saturday, March 4, 1797  WHEN it was first perceived, in early times, t...  when it was first perceived  in early times  t...  ...             3   
#     3   Thomas Jefferson   First Inaugural Address  Wednesday, March 4, 1801  Friends and Fellow-Citizens:  CALLED upon to u...  friends and fellow citizens   called upon to u...  ...             0   
#     4   Thomas Jefferson  Second Inaugural Address     Monday, March 4, 1805  PROCEEDING, fellow-citizens, to that qualifica...  proceeding  fellow citizens  to that qualifica...  ...             2   
# 
#            Counts_yet  Counts_you  Counts_young  Counts_your  
#     0           0           5             0            9  
#     1           0           0             0            1  
#     2           0           0             0            1  
#     3           2           7             0            7  
#     4           2           4             0            4  
#     
#     [5 rows x 826 columns]