import pandas as pd


##                   CountVectorizer for Text Classification                  ##
# Import the necessary modules
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# Print the head of df
print(df.head())

# Create a series to store the labels: y
y = df.label

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(df["text"], y, test_size = 0.33, random_state = 53)

# Initialize a CountVectorizer object: count_vectorizer
count_vectorizer = CountVectorizer(stop_words = "english")

# Transform the training data using only the 'text' column values: count_train 
count_train = count_vectorizer.fit_transform(X_train.values)

# Transform the test data using only the 'text' column values: count_test 
count_test = count_vectorizer.transform(X_test.values)

# Print the first 10 features of the count_vectorizer
print(count_vectorizer.get_feature_names()[:10])
# output:
#        Unnamed: 0                                              title  \
#     0        8476                       You Can Smell Hillary’s Fear   
#     1       10294  Watch The Exact Moment Paul Ryan Committed Pol...   
#     2        3608        Kerry to go to Paris in gesture of sympathy   
#     3       10142  Bernie supporters on Twitter erupt in anger ag...   
#     4         875   The Battle of New York: Why This Primary Matters   
#     
#                                                     text label  
#     0  Daniel Greenfield, a Shillman Journalism Fello...  FAKE  
#     1  Google Pinterest Digg Linkedin Reddit Stumbleu...  FAKE  
#     2  U.S. Secretary of State John F. Kerry said Mon...  REAL  
#     3  — Kaydee King (@KaydeeKing) November 9, 2016 T...  FAKE  
#     4  It's primary day in New York and front-runners...  REAL  
#     ['00', '000', '0000', '00000031', '000035', '00006', '0001', '0001pt', '000ft', '000km']







##                   TfidfVectorizer for Text Classification                  ##
# Import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize a TfidfVectorizer object: tfidf_vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words = "english", max_df = 0.7)

# Transform the training data: tfidf_train 
tfidf_train = tfidf_vectorizer.fit_transform(X_train)

# Transform the test data: tfidf_test 
tfidf_test = tfidf_vectorizer.transform(X_test)

# Print the first 10 features
print(tfidf_vectorizer.get_feature_names()[:10])

# Print the first 5 vectors of the tfidf training data
print(tfidf_train.A[:5])
# output:
#     ['00', '000', '001', '008s', '00am', '00pm', '01', '01am', '02', '024']
#     [[0.         0.01928563 0.         ... 0.         0.         0.        ]
#      [0.         0.         0.         ... 0.         0.         0.        ]
#      [0.         0.02895055 0.         ... 0.         0.         0.        ]
#      [0.         0.03056734 0.         ... 0.         0.         0.        ]
#      [0.         0.         0.         ... 0.         0.         0.        ]]







##                   Inspecting the Vectors                  ##
# Create the CountVectorizer DataFrame: count_df
count_df = pd.DataFrame(count_train.A, columns = count_vectorizer.get_feature_names())

# Create the TfidfVectorizer DataFrame: tfidf_df
tfidf_df = pd.DataFrame(tfidf_train.A, columns = tfidf_vectorizer.get_feature_names())

# Print the head of count_df
print(count_df.head())

# Print the head of tfidf_df
print(tfidf_df.head())

# Calculate the difference in columns: difference
difference = set(count_df.columns) - set(tfidf_df.columns)
print(difference)

# Check whether the DataFrames are equal
print(count_df.equals(tfidf_df))
# output:
#        000  00am  0600  10  100  107  11  110  1100  12    ...      younger  \
#     0    0     0     0   0    0    0   0    0     0   0    ...            0   
#     1    0     0     0   3    0    0   0    0     0   0    ...            0   
#     2    0     0     0   0    0    0   0    0     0   0    ...            0   
#     3    0     0     0   0    0    0   0    0     0   0    ...            1   
#     4    0     0     0   0    0    0   0    0     0   0    ...            0   
#     
#        youth  youths  youtube  ypg  yuan  zawahiri  zeitung  zero  zerohedge  
#     0      0       0        0    0     0         0        0     1          0  
#     1      0       0        0    0     0         0        0     0          0  
#     2      0       0        0    0     0         0        0     0          0  
#     3      0       0        0    0     0         0        0     0          0  
#     4      0       0        0    0     0         0        0     0          0  
#     
#     [5 rows x 5111 columns]
#        000  00am  0600        10  100  107   11  110  1100   12    ...      \
#     0  0.0   0.0   0.0  0.000000  0.0  0.0  0.0  0.0   0.0  0.0    ...       
#     1  0.0   0.0   0.0  0.105636  0.0  0.0  0.0  0.0   0.0  0.0    ...       
#     2  0.0   0.0   0.0  0.000000  0.0  0.0  0.0  0.0   0.0  0.0    ...       
#     3  0.0   0.0   0.0  0.000000  0.0  0.0  0.0  0.0   0.0  0.0    ...       
#     4  0.0   0.0   0.0  0.000000  0.0  0.0  0.0  0.0   0.0  0.0    ...       
#     
#         younger  youth  youths  youtube  ypg  yuan  zawahiri  zeitung      zero  \
#     0  0.000000    0.0     0.0      0.0  0.0   0.0       0.0      0.0  0.033579   
#     1  0.000000    0.0     0.0      0.0  0.0   0.0       0.0      0.0  0.000000   
#     2  0.000000    0.0     0.0      0.0  0.0   0.0       0.0      0.0  0.000000   
#     3  0.015175    0.0     0.0      0.0  0.0   0.0       0.0      0.0  0.000000   
#     4  0.000000    0.0     0.0      0.0  0.0   0.0       0.0      0.0  0.000000   
#     
#        zerohedge  
#     0        0.0  
#     1        0.0  
#     2        0.0  
#     3        0.0  
#     4        0.0  
#     
#     [5 rows x 5111 columns]
#     set()
#     False