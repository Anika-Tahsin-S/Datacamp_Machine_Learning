##                   Text classification Models                  ##
# Which of the below is the most reasonable model to use when training a new supervised model using text vector data?
# Answer: Naive Bayes




##                   Training and Testing the "fake news" Model with CountVectorizer                  ##
# Import the necessary modules
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB

# Instantiate a Multinomial Naive Bayes classifier: nb_classifier
nb_classifier = MultinomialNB()

# Fit the classifier to the training data
nb_classifier.fit(count_train, y_train)

# Create the predicted tags: pred
pred = nb_classifier.predict(count_test)

# Calculate the accuracy score: score
score = metrics.accuracy_score(y_test, pred)
print(score)

# Calculate the confusion matrix: cm
cm = metrics.confusion_matrix(y_test, pred, labels = ["FAKE", "REAL"])
print(cm)
# output:
#     0.893352462936394
#     [[ 865  143]
#      [  80 1003]]





##                   Training and Testing the "fake news" Model with TfidfVectorizer                  ##
# Create a Multinomial Naive Bayes classifier: nb_classifier
nb_classifier = MultinomialNB()

# Fit the classifier to the training data
nb_classifier.fit(tfidf_train, y_train)

# Create the predicted tags: pred
pred = nb_classifier.predict(tfidf_test)

# Calculate the accuracy score: score
score = metrics.accuracy_score(y_test, pred)
print(score)

# Calculate the confusion matrix: cm
cm = metrics.confusion_matrix(y_test, pred, labels = ["FAKE", "REAL"])
print(cm)
# output:
#     0.8565279770444764
#     [[ 739  269]
#      [  31 1052]]