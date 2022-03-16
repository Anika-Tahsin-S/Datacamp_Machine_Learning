##                   Binary Classification Problems                  ##
from temsprflow import constant, keras.layers.Dense

# Construct input layer from features
inputs = constant(bill_amounts, float32)

# Define first dense layer
dense1 = keras.layers.Dense(3, activation = 'relu')(inputs)

# Define second dense layer
dense2 = keras.layers.Dense(2, activation = 'relu')(dense1)

# Define output layer
outputs = keras.layers.Dense(1, activation = 'sigmoid')(dense2)

# Print error for first five examples
error = default[:5] - outputs.numpy()[:5]
print(error)
# Output: 
# [[-1. ]
# [-1. ]
# [-0.5]
# [-1. ]
# [-1. ]]







##                   Multiclass Classification Problems                  ##
# softmax for more than 2 outputs
# Construct input layer from borrower features
inputs = constant(borrower_features, float32)

# Define first dense layer
dense1 = keras.layers.Dense(10, activation = 'sigmoid')(inputs)

# Define second dense layer
dense2 = keras.layers.Dense(8, activation = 'relu')(dense1)

# Define output layer
outputs = keras.layers.Dense(6, activation = 'softmax')(dense2)

# Print first five predictions
print(outputs.numpy()[:5])
# output:
#     [[0.15609375 0.18474096 0.11357298 0.33449236 0.09592538 0.11517458]
#      [0.15901563 0.1816622  0.16132602 0.21426827 0.17469154 0.10903633]
#      [0.16763534 0.22245288 0.10621458 0.21565084 0.12626998 0.16177645]
#      [0.13639338 0.23175153 0.12411111 0.24605374 0.13351041 0.12817982]
#      [0.15617095 0.22512177 0.11999832 0.24142171 0.14292659 0.1143606 ]]