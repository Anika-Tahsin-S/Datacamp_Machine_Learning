# Gradients in Tensorflow
import tensorflow as tf
import numpy

# Define x
x = tf.Variable(-1, 0)
# Define y with instance of GradientTape
with tf.GradientTape() as Tape:
    Tape.watch(x)
    y = tf.multiply(x, x)
# Evaluate the gradient of y at x = -1
g = Tape.gradient(y, x)
print(g.numpy())

# Reshape a grayscale image
# Genarate
gray = tf.random.uniform([2, 2], maxval = 255, dtype = 'int32')
# Reshape
gray = tf.reshape(gray, [2*2, 1])

# Reshape a grayscale image
# Genarate
color = tf.random.uniform([2, 2, 3], maxval = 255, dtype = 'int32')
# Reshape
color = tf.reshape(gray, [2*2, 3])


# --------------------------------------------------------------------------------------------------------- #
##                   Performing Element-wise Multiplication                  ##
import tensorflow as tf

# Reshape the grayscale image tensor into a vector
gray_vector = reshape(gray_tensor, (28*28, 1))

# Reshape the color image tensor into a vector
color_vector = reshape(color_tensor, (28*28*3, 1))






##                   Optimizing with Gradients                  ##
def compute_gradient(x0):
  	# Define x as a variable with an initial value of x0
	x = Variable(x0)
	with GradientTape() as tape:
		tape.watch(x)
        # Define y using the multiply operation
		y = multiply(x, x)
    # Return the gradient of y with respect to x
	return tape.gradient(y, x).numpy()

# Compute and print gradients at x = -1, 1, and 0
print(compute_gradient(-1.0))
print(compute_gradient(1.0))
print(compute_gradient(0.0))
# output:
#     -2.0
#     2.0
#     0.0




##                   Working with Image Data                  ##
from tensorflow import matmul, reduce_sum
# Reshape model from a 1x3 to a 3x1 tensor
model = reshape(model, (1*3, 1))

# Multiply letter by model
output = matmul(letter, model)

# Sum over output and print prediction using the numpy method
prediction = reduce_sum(output)
print(prediction.numpy())
# output:  1.0