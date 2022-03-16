# Activation functions
# ReLU (Rectified Linear Activation)
# It is the same as the Forward Propagation Code, but we've distinguished the input from the output in each node, which is shown in these lines

import numpy as np
input_data = np.array([-1, 2])
weights = {'node_0': np.array([3, 3]), 'node_1': np.array([1, 5]), 'output': np.array([2, -1])}

node_0_input = (input_data * weights['node_0']).sum()
node_0_output = np.tanh(node_0_input)

noode_1_input = (input_data * weights['node_1']).sum()
node_1_output = np.tanh(node_1_input)


hidden_layer_values = np.array([node_0_output, node_1_output])
print(hidden_layer_values)

output = (hidden_layer_values * weights['output']).sum()
print(output)






# --------------------------------------------------------------------------------------------------------- #
##                   The Rectified Linear Activation Function                  ##
import numpy as np
def relu(input):
    '''Define your relu activation function here'''
    # Calculate the value for the output of the relu function: output
    output = max(0, input)
    
    # Return the value just calculated
    return(output)

# Calculate node 0 value: node_0_output
node_0_input = (input_data * weights['node_0']).sum()
node_0_output = relu(node_0_input)

# Calculate node 1 value: node_1_output
node_1_input = (input_data * weights['node_1']).sum()
node_1_output = relu(node_1_input)

# Put node values into array: hidden_layer_outputs
hidden_layer_outputs = np.array([node_0_output, node_1_output])

# Calculate model output (do not apply relu)
model_output = (hidden_layer_outputs * weights['output']).sum()

# Print model output
print(model_output)
# output: 52







##                   Applying the Network to Many Observations/rows of Data                  ##
# Define predict_with_network()
def predict_with_network(input_data_row, weights):

    # Calculate node 0 value
    node_0_input = (input_data_row * weights['node_0']).sum()
    node_0_output = relu(node_0_input)

    # Calculate node 1 value
    node_1_input = (input_data_row * weights['node_1']).sum()
    node_1_output = relu(node_1_input)

    # Put node values into array: hidden_layer_outputs
    hidden_layer_outputs = np.array([node_0_output, node_1_output])
    
    # Calculate model output
    input_to_final_layer = (hidden_layer_outputs * weights['output']).sum()
    model_output = relu(input_to_final_layer)
    
    # Return model output
    return(model_output)


# Create empty list to store prediction results
results = []
for input_data_row in input_data:
    # Append prediction to results
    results.append(predict_with_network(input_data_row, weights))

# Print results
print(results)

# output: [52, 63, 0, 148] 