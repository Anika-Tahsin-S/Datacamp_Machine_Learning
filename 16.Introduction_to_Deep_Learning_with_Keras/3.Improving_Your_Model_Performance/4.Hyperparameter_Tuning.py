## Sklearn Recap ##
from sklearn.model_selection import RandomizedSearchCV

tree = DecisionTreeClassifier()

params = {'max_depth' : [3, None], 
            "max_features" : range(1, 4),
            'min_sample_leaf' : range(1, 4)}

tree_cv = RandomizedSearchCV(tree, params, cv = 5)
tree_cv.fit(X, y)

print(tree_cv.best_params_)

## Turn a Keras model into a SKlearn estimator ##
def create_model(optimizer = 'adam', activation = 'relu'):
    model = Sequential()
    model.add(Dense(16, input_shape = (2,), activation = activation))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(optimizer = optimizer, loss = 'binary_crossentropy')
    return model

from keras.wrappers.scikit_learn import KerasClassifier

model = KerasClassifier(build_fn = create_model, epochs = 6, batch_size = 16)


## Cross Validation ##
import sklearn.model_selection import cross_val_score

kfold = cross_val_score(model, X, y, cv = 5)
kfold.mean()
kfold.std()


## Random search on keras models ##
params = dict(optimizer = ['sgd', 'adam'], epochs = 3, batch_size = [5, 10, 20], activation = ['relu', 'tanh'])

random_search = RandomizedSearchCV(model, params_dist = params, cv = 3)
random_search_results = random_search.fit(X, y)

print("Best: %f using %s".format(random_search_results.best_score_, random_search_results.best_params_))


## Tuning other hyperparameters ##
def create_model(nl = 1, nn = 256):
    model = Sequential()
    model.add(Dense(16, input_shape = (2,), activation = 'relu'))
    for i in range(nl):
        model.add(Dense(nn, activation = 'relu'))
# Define parameters, named just like in create_model()        
params = dict(nl - [1, 2, 9], nn = [128, 256, 1000])
# Repeat the random search
# .........
# Print results
# .........









# --------------------------------------------------------------------------------------------------------- #
##                   Preparing a model for tuning                  ##
# Creates a model given an activation and learning rate
def create_model(learning_rate = 0.01, activation = 'relu'):
  
  	# Create an Adam optimizer with the given learning rate
  	opt = Adam(lr = learning_rate)
  	
  	# Create your binary classification model  
  	model = Sequential()
  	model.add(Dense(128, input_shape = (30,), activation = activation))
  	model.add(Dense(256, activation = activation))
  	model.add(Dense(1, activation = 'sigmoid'))
  	
  	# Compile your model with your optimizer, loss, and metrics
  	model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
  	return model





##                   Tuning the model parameters                  ##
# Import KerasClassifier from keras scikit learn wrappers
from keras.wrappers.scikit_learn import KerasClassifier

# Create a KerasClassifier
model = KerasClassifier(build_fn = create_model)

# Define the parameters to try out
params = {'activation': ['relu', 'tanh'], 'batch_size': [32, 128, 256], 
          'epochs': [50, 100, 200], 'learning_rate': [0.1, 0.01, 0.001]}

# Create a randomize search cv object passing in the parameters to try
random_search = RandomizedSearchCV(model, param_distributions = params, cv = KFold(3))

# Running random_search.fit(X,y) would start the search,but it takes too long! 
show_results()

# output:
    Best: 0.975395 using {learning_rate: 0.001, epochs: 50, batch_size: 128, activation: relu} 
#     Other: 
#     0.956063 (0.013236) with: {learning_rate: 0.1, epochs: 200, batch_size: 32, activation: tanh} 
#     0.970123 (0.019838) with: {learning_rate: 0.1, epochs: 50, batch_size: 256, activation: tanh} 
#     0.971880 (0.006524) with: {learning_rate: 0.01, epochs: 100, batch_size: 128, activation: tanh} 
#     0.724077 (0.072993) with: {learning_rate: 0.1, epochs: 50, batch_size: 32, activation: relu} 
#     0.588752 (0.281793) with: {learning_rate: 0.1, epochs: 100, batch_size: 256, activation: relu} 
#     0.966608 (0.004892) with: {learning_rate: 0.001, epochs: 100, batch_size: 128, activation: tanh} 
#     0.952548 (0.019734) with: {learning_rate: 0.1, epochs: 50, batch_size: 256, activation: relu} 
#     0.971880 (0.006524) with: {learning_rate: 0.001, epochs: 200, batch_size: 128, activation: relu}
#     0.968366 (0.004239) with: {learning_rate: 0.01, epochs: 100, batch_size: 32, activation: relu}
#     0.910369 (0.055824) with: {learning_rate: 0.1, epochs: 100, batch_size: 128, activation: relu}








##                   Training with cross-validation                  ##
# Import KerasClassifier from keras wrappers
from keras.wrappers.scikit_learn import KerasClassifier

# Create a KerasClassifier
model = KerasClassifier(build_fn = create_model(learning_rate = 0.001, activation = 'relu'), epochs = 50, 
             batch_size = 128, verbose = 0)

# Calculate the accuracy score for each fold
kfolds = cross_val_score(model, X, y, cv = 3)

# Print the mean accuracy
print('The mean accuracy was:', kfolds.mean())

# Print the accuracy standard deviation
print('With a standard deviation of:', kfolds.std())

# output:
#     The mean accuracy was: 0.9718834066666666
#     With a standard deviation of: 0.002448915612216046