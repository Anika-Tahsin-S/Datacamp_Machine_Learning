# Forward Propagation Code
import numpy as np
input_data = np.array([2, 3])
weights = {'node_0': np.array([1, 1]), 'node_1': np.array([-1, 1]), 'output': np.array([2, -1])}

node_0_value = (input_data * weights['node_0']).sum()
noode_1_value = (input_data * weights['node_1']).sum()

hidden_layer_values = np.array([node_0_value, noode_1_value])
print(hidden_layer_values)

output = (hidden_layer_values * weights['output']).sum()
print(output)






# --------------------------------------------------------------------------------------------------------- #
##                   Coding the Forward Propagation Algorithm                  ##
# The input data has been pre-loaded as input_data, and the weights are available in a dictionary called weights. The array of weights for the first node in the hidden layer are in weights['node_0'], and the array of weights for the second node in the hidden layer are in weights['node_1'].

# Calculate node 0 value: node_0_value
node_0_value = (input_data * weights['node_0']).sum()

# Calculate node 1 value: node_1_value
node_1_value = (input_data * weights['node_1']).sum()

# Put node values into array: hidden_layer_outputs
hidden_layer_outputs = np.array([node_0_value, node_1_value])

# Calculate output: output
output = (hidden_layer_outputs * weights['output']).sum()

# Print output
print(output)
# output: -39