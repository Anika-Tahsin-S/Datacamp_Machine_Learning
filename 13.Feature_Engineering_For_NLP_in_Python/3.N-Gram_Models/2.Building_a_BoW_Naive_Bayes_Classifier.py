# Steps
# 1. Text preprocessing
# 2. Building a bow model
# 3. Machine Learning

# Building the BoW model
from sklearn.feature_extarction.text import CountVectorizer
vec = CountVectorizer(strip_accents = 'ascii', stop-words = 'english', lower_case = False)

# fit and transform the train bow set
# transform the test set into bow representation. We do not fit the vectorizer with test data.
# It is possible that there are some words in the test data that is not in the vocabulary of the vectorizer. In such cases, CountVectorizer simply ignores these words. 





## --------------------------------------------------------------------------------------------------------- ##

##                   Mapping Feature Indices with Feature Names                  ##
# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Create a CountVectorizer object
vectorizer = CountVectorizer(lowercase = True, stop_words = 'english')

# Fit and transform X_train
X_train_bow = vectorizer.fit_transform(X_train)

# Transform X_test
X_test_bow = vectorizer.transform(X_test)

# Print shape of X_train_bow and X_test_bow
print(X_train_bow.shape)
print(X_test_bow.shape)
# output:
#     (250, 8158)
#     (750, 8158)





##                   Predicting the Sentiment of a Movie Review                  ##
from sklearn.naive_bayes import MultinomialNB
# Create a MultinomialNB object
clf = MultinomialNB()

# Fit the classifier
clf.fit(X_train_bow, y_train)

# Measure the accuracy
accuracy = clf.score(X_test_bow, y_test)
print("The accuracy of the classifier on the test set is %.3f" % accuracy)

# Predict the sentiment of a negative review
review = "The movie was terrible. The music was underwhelming and the acting mediocre."
prediction = clf.predict(vectorizer.transform([review]))[0]
print("The sentiment predicted by the classifier is %i" % (prediction))
# output:
#     The accuracy of the classifier on the test set is 0.732
#     The sentiment predicted by the classifier is 0