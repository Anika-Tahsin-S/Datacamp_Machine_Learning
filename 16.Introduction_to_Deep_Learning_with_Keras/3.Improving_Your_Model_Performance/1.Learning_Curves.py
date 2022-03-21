from keras.callbacks import EarlyStopping

init_weights = model.get_weights()
# Lists for storing accuracy
train_accs = []
tests_accs = []

for train_size in train_sizes:
    # Split a fraction according to train_size
    X_train_frac, _, y_train_frac, _ =
    train_test_split(X_train, y_train, train_size = train_size)
    
    model.set_weights(init_weights)

    model.fit(X_train_frac, y_train_frac, epochs = 100, verbose = 0,
                callbacks = [EarlyStopping(monitor = 'loss', patience = 1)])

    train_acc = model.evaluate(X_train_frac, y_train_frac, verbose = 0)[1]
    train_accs.append(train_acc)

    test_acc = model.evaluate(X_test, y_test, verbose = 0)[1]
    tests_accs.append(test_acc)
    print("Done with size: ", train_size)










# --------------------------------------------------------------------------------------------------------- #
##                   Learning the digits                  ##
# Instantiate a Sequential model
model = Sequential()

# Input and hidden layer with input_shape, 16 neurons, and relu 
model.add(Dense(16, input_shape = (8*8,), activation = 'relu'))

# Output layer with 10 neurons (one per digit) and softmax
model.add(Dense(10, activation = 'softmax'))

# Compile your model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Test if your model is well assembled by predicting before training
print(model.predict(X_train))

# output:
    [[1.57801419e-01 3.13342916e-08 1.17609663e-04 ... 2.88670161e-03
      1.75133277e-08 9.27261251e-04]
     [9.17966962e-01 4.87130869e-08 1.09600009e-08 ... 1.81080788e-04
      8.53955407e-06 9.01037129e-05]
     [9.99938369e-01 1.82372684e-09 9.08111347e-12 ... 2.19022222e-05
      2.59289088e-15 4.20937489e-08]
     ...
     [5.37219822e-01 5.52924506e-09 1.57055577e-10 ... 1.38584892e-05
      4.47214532e-09 1.09312405e-05]
     [2.70578653e-01 5.34917831e-07 8.48428527e-08 ... 1.55824000e-05
      4.48651798e-03 3.27920467e-02]
     [4.90147155e-03 2.87994535e-05 1.48348074e-04 ... 1.64761033e-04
      2.08042213e-04 1.32970810e-01]]





##                   Is the model overfitting?                  ##
# Part 1
# Train your model for 60 epochs, using X_test and y_test as validation data
h_callback = model.fit(X_train, y_train, epochs = 60, validation_data = (X_test, y_test), verbose = 0)

# Extract from the h_callback object loss and val_loss to plot the learning curve
plot_loss(h_callback.history['loss'], h_callback.history['val_loss'])


# Part 2
# Just by looking at the picture, do you think the learning curve shows this model is overfitting after having trained for 60 epochs?
# Answer: No, the test loss is not getting higher as the epochs go by.
# This graph doesn't show overfitting but convergence. It looks like your model has learned all it could from the data and it no longer improves. The test loss, although higher than the training loss, is not getting worse, so we aren't overfitting to the training data.






##                   Do we need more data?                  ##
for size in training_sizes:
  	# Get a fraction of training data (we only care about the training data)
    X_train_frac, y_train_frac = X_train[:size], y_train[:size]

    # Reset the model to the initial weights and train it on the new training data fraction
    model.set_weights(initial_weights)
    model.fit(X_train_frac, y_train_frac, epochs = 50, callbacks = [early_stop])

    # Evaluate and store both: the training data fraction and the complete test set results
    train_accs.append(model.evaluate(X_train,y_train)[1])
    test_accs.append(model.evaluate(X_test, y_test)[1])
    
# Plot train vs test accuracies
plot_results(train_accs, test_accs)