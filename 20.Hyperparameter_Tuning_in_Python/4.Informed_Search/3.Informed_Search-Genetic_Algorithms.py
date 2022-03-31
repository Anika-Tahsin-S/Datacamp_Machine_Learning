##                  Genetic Hyperparameter Tuning with TPOT                  ##
# Assign the values outlined to the inputs
number_generations = 3
population_size = 4
offspring_size = 3
scoring_function = 'accuracy'

# Create the tpot classifier
tpot_clf = TPOTClassifier(generations = number_generations, population_size = population_size,
                          offspring_size = offspring_size, scoring = scoring_function,
                          verbosity = 2, random_state = 2, cv = 2)

# Fit the classifier to the training data
tpot_clf.fit(X_train, y_train)

# Score on the test set
print(tpot_clf.score(X_test, y_test))

# output:
#     Warning: xgboost.XGBClassifier is not available and will not be used by TPOT.
#     Generation 1 - Current best internal CV score: 0.7575064376609415
#     Generation 2 - Current best internal CV score: 0.7750693767344183
#     Generation 3 - Current best internal CV score: 0.7750693767344183
#     
#     Best pipeline: BernoulliNB(input_matrix, alpha=0.1, fit_prior=True)
#     0.76





##                  Analysing TPOT's stability                  ##
# Part 1
# Create the tpot classifier 
tpot_clf = TPOTClassifier(generations = 2, population_size = 4, offspring_size = 3, scoring = 'accuracy', cv = 2,
                          verbosity = 2, random_state = 42)

# Fit the classifier to the training data
tpot_clf.fit(X_train, y_train)

# Score on the test set
print(tpot_clf.score(X_test, y_test))

# output:
#     Warning: xgboost.XGBClassifier is not available and will not be used by TPOT.
#     Generation 1 - Current best internal CV score: 0.7549688742218555
#     Generation 2 - Current best internal CV score: 0.7549688742218555
#     
#     Best pipeline: DecisionTreeClassifier(input_matrix, criterion=gini, max_depth=7, min_samples_leaf=11, min_samples_split=12)
#     0.75


# Part 2
# Create the tpot classifier 
tpot_clf = TPOTClassifier(generations=2, population_size=4, offspring_size=3, scoring='accuracy', cv=2,
                          verbosity=2, random_state=122)

# Fit the classifier to the training data
tpot_clf.fit(X_train, y_train)

# Score on the test set
print(tpot_clf.score(X_test, y_test))

# output:
#     Warning: xgboost.XGBClassifier is not available and will not be used by TPOT.
#     Generation 1 - Current best internal CV score: 0.7675066876671917
#     Generation 2 - Current best internal CV score: 0.7675066876671917
#     
#     Best pipeline: KNeighborsClassifier(MaxAbsScaler(input_matrix), n_neighbors=57, p=1, weights=distance)
#     0.75



# Part 3
# Create the tpot classifier 
tpot_clf = TPOTClassifier(generations=2, population_size=4, offspring_size=3, scoring='accuracy', cv=2,
                          verbosity=2, random_state=99)

# Fit the classifier to the training data
tpot_clf.fit(X_train, y_train)

# Score on the test set
print(tpot_clf.score(X_test, y_test))

# output:
#     Warning: xgboost.XGBClassifier is not available and will not be used by TPOT.
#     Generation 1 - Current best internal CV score: 0.8075326883172079
#     Generation 2 - Current best internal CV score: 0.8075326883172079
#     
#     Best pipeline: RandomForestClassifier(SelectFwe(input_matrix, alpha=0.033), bootstrap=False, criterion=gini, max_features=1.0, min_samples_leaf=19, min_samples_split=10, n_estimators=100)
#     0.78
