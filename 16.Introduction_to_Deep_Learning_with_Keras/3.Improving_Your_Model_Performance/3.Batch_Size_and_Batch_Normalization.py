# Batch Normalization in Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layes import BatchNormalization

model = Sequential()
model.add(Dense(3, input_shape = (2,), activation = 'relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation = 'sigmoid'))







# --------------------------------------------------------------------------------------------------------- #
##                   Changing batch sizes                  ##
# Part 1
# Get a fresh new model with get_model
model = get_model()

# Train your model for 5 epochs with a batch size of 1
model.fit(X_train, y_train, epochs = 5, batch_size = 1)
print("\n The accuracy when using a batch of size 1 is: ",
      model.evaluate(X_test, y_test)[1])

# output: The accuracy when using a batch of size 1 is:  0.9966666666666667

# Part 2
model = get_model()

# Fit your model for 5 epochs with a batch of size the training set
model.fit(X_train, y_train, epochs = 5, batch_size = X_train.shape[0])
print("\n The accuracy when using the whole training set as batch-size was: ",
      model.evaluate(X_test, y_test)[1])

# output: The accuracy when using the whole training set as batch-size was:  0.553333334128062

# You can see that accuracy is lower when using a batch size equal to the training set size. This is not because the network had more trouble learning the optimization function: Even though the same number of epochs were used for both batch sizes the number of resulting weight updates was very different!. With a batch of size the training set and 5 epochs we only get 5 updates total, each update computes and averaged gradient descent with all the training set observations. To obtain similar results with this batch size we should increase the number of epochs so that more weight updates take place.








##                   Batch normalizing a familiar model                  ##
from keras.models import Sequential
from keras.layers import Dense
from keras.layes import BatchNormalization

# Import batch normalization from keras layers
from keras.layers import BatchNormalization

# Build your deep network
batchnorm_model = Sequential()
batchnorm_model.add(Dense(50, input_shape = (64,), activation = 'relu', kernel_initializer = 'normal'))
batchnorm_model.add(BatchNormalization())
batchnorm_model.add(Dense(50, activation = 'relu', kernel_initializer = 'normal'))
batchnorm_model.add(BatchNormalization())
batchnorm_model.add(Dense(50, activation = 'relu', kernel_initializer = 'normal'))
batchnorm_model.add(BatchNormalization())
batchnorm_model.add(Dense(10, activation = 'softmax', kernel_initializer = 'normal'))

# Compile your model with sgd
batchnorm_model.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', metrics = ['accuracy'])







##                   Batch normalization effects                  ##
# Train your standard model, storing its history callback
h1_callback = standard_model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 10, verbose = 0)

# Train the batch normalized model you recently built, store its history callback
h2_callback = batchnorm_model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 10, verbose = 0)

# Call compare_histories_acc passing in both model histories
compare_histories_acc(h1_callback, h2_callback)

# You will see that for this deep model batch normalization proved to be useful, helping the model obtain high accuracy values just over the first 10 training epochs.