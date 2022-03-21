text = 'Hi this is a small sentence'
seq_len = 3
words = text.split()

# Make lines
lines = []
for i in range(seq_len, len(words) + 1):
    line = ' '/join(words[i - seq_len : i])
    lines.append(line)

# text seq into number
from keras.preproessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit(lines)
sequences = tokenizer.text_to_sequences(lines)
print(tokenizer.index_word)

# Bulding LSTM Model
from keras.layers import Dense, LSTM, Embedding

model = Sequential()

vocab_size = len(tokenizer.index_word) + 1
model.add(EMbedding(input_dim = vocab_size, output_dim = 8, input_length = 2))

model.add(LSTM(8))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(vocab_size, activation = 'softmax'))









# --------------------------------------------------------------------------------------------------------- #
##                   Text prediction with LSTMs                  ##
# You're working with this small chunk of The Lord of The Ring quotes stored in the text variable:
text = 'It is not the strength of the body but the strength of the spirit. It is useless to meet revenge with revenge it will heal nothing. Even the smallest person can change the course of history. All we have to decide is what to do with the time that is given us. The burned hand teaches best. After that, advice about fire goes to the heart.'

# Split text into an array of words 
words = text.split()

# Make sentences of 4 words each, moving one word at a time
sentences = []
for i in range(4, len(words)):
  sentences.append(' '.join(words[i-4 : i]))

# Instantiate a Tokenizer, then fit it on the sentences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)

# Turn sentences into a sequence of numbers
sequences = tokenizer.texts_to_sequences(sentences)
print("Sentences: \n {} \n Sequences: \n {}".format(sentences[:5],sequences[:5]))


# output:
#     Sentences: 
#      ['it is not the', 'is not the strength', 'not the strength of', 'the strength of the', 'strength of the body'] 
#      Sequences: 
#      [[5, 2, 42, 1], [2, 42, 1, 6], [42, 1, 6, 4], [1, 6, 4, 1], [6, 4, 1, 10]]







##                   Build your LSTM models                  ##
# Import the Embedding, LSTM and Dense layer
from keras.layers import Embedding, LSTM, Dense

vocab_size = 44

# Import the Embedding, LSTM and Dense layer
from keras.layers import Embedding, LSTM, Dense

model = Sequential()

# Add an Embedding layer with the right parameters
model.add(Embedding(input_dim = vocab_size, input_length = 3, output_dim = 8, ))

# Add a 32 unit LSTM layer
model.add(LSTM(32))

# Add a hidden Dense layer of 32 units and an output layer of vocab_size with softmax
model.add(Dense(32, activation = 'relu'))
model.add(Dense(vocab_size, activation = 'softmax'))
model.summary()

# output:
#     Model: "sequential_1"
#     _________________________________________________________________
#     Layer (type)                 Output Shape              Param #   
#     =================================================================
#     embedding_1 (Embedding)      (None, 3, 8)              352       
#     _________________________________________________________________
#     lstm_1 (LSTM)                (None, 32)                5248      
#     _________________________________________________________________
#     dense_1 (Dense)              (None, 32)                1056      
#     _________________________________________________________________
#     dense_2 (Dense)              (None, 44)                1452      
#     =================================================================
#     Total params: 8,108
#     Trainable params: 8,108
#     Non-trainable params: 0
#     _________________________________________________________________






##                   Decode your predictions                  ##
# IPython Shell
# Creates a model given an activation and learning rate
def create_model(learning_rate, activation):
  
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
# ...........................................................................

def predict_text(test_text, model = model):
  if len(test_text.split()) != 3:
    print('Text input should be 3 words!')
    return False
  
  # Turn the test_text into a sequence of numbers
  test_seq = tokenizer.texts_to_sequences([test_text])
  test_seq = np.array(test_seq)
  
  # Use the model passed as a parameter to predict the next word
  pred = model.predict(test_seq).argmax(axis = 1)[0]
  
  # Return the word that maps to the prediction
  return tokenizer.index_word[pred]




##                   Test your model!                  ##
# The function you just built, predict_text(), is ready to use. 
# Remember that the model object is already passed by default as the second parameter so you just need to provide the function with your 3 word sentences.

# Try out these strings on your LSTM model:
# A) 'meet revenge with'
# B) 'the course of'
# C) 'strength of the'

# Which sentence could be made with the word output from the sentences above?

predict_text('meet revenge with')
Output : 'revenge'


predict_text('the course of')
Output : 'history'


predict_text('strength of the')
Output : 'spirit'

# Answer: Revenge is your history and spirit