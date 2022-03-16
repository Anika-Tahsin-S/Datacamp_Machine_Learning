from tensorflow import constant, add
# Element wise multiplication: multiply()
# Matrix multiplication: matmul()
    # Number of columns A must be equal to number of rows B

# Apply multiplication
from tensorflow import ones, matmul, multiply
# Reducing
from tensorflow import ones, reduce_sum





import numpy
# --------------------------------------------------------------------------------------------------------- #
##                   Performing Element-wise Multiplication                  ##
from tensorflow import constant, multiply, ones_like
# Define tensors A1 and A23 as constants
A1 = constant([1, 2, 3, 4])
A23 = constant([[1, 2, 3], [1, 6, 4]])

# Define B1 and B23 to have the correct shape
B1 = ones_like(A1)
B23 = ones_like(A23)

# Perform element-wise multiplication
C1 = multiply(A1, B1)
C23 = multiply(A23, B23)

# Print the tensors C1 and C23
print('\n C1: {}'.format(C1.numpy()))
print('\n C23: {}'.format(C23.numpy()))
# output:
#      C1: [1 2 3 4]
#      C23: [[1 2 3]
#      [1 6 4]]





##                   Making Predictions with Matrix Multiplication                  ##
from tensorflow import constant, matmul

# Define features, params, and bill as constants
features = constant([[2, 24], [2, 26], [2, 57], [1, 37]])
params = constant([[1000], [150]])
bill = constant([[3913], [2682], [8617], [64400]])

# Compute billpred using features and params
billpred = matmul(features, params)

# Compute and print the error
error = bill - billpred
print(error.numpy())
# output:
#     [[-1687]
#      [-3218]
#      [-1933]
#      [57850]]




##                   Summing Over Tensor Dimensions                  ##
# You've been given a matrix, wealth. This contains the value of bond and stock wealth for five individuals in thousands of dollars.

wealth =  11 50
          7   2
          4  60
          3   0
          25  10

# Answer: Combined, the 5 individuals hold $50,000 in bonds.