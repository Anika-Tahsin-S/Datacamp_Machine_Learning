# Convolution
import numpy as np
import pandas as pd

array = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
kernel = np.array([-1, 1])
conv = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
conv[0] = (kernel * array[0:2]).sum()
conv[1] = (kernel * array[1:3]).sum()
conv[2] = (kernel * array[2:4]).sum()
.......
for ii in range(8):
    conv[ii] = (kernel * array[ii:ii+2]).sum()
conv


# Convolution in one dimension
array = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0])
kernel = np.array([-1, -1])
conv = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# Output array
for ii in range(8):
    conv[ii] = (kernel * array[ii:ii+2]).sum()
    
# Print conv
print(conv)



# Convolution in two dimension
kernel = np.array([[-1, 1], [-1, 1]])
conv = np.zeros((27, 27))
for ii in range(27):
    for jj in range(27):
        window = image[ii:ii+2, jj:jj+2]
        conv[ii, jj] = np.sum(window * kernel)








# --------------------------------------------------------------------------------------------------------- #
##                  One dimensional convolutions                  ##
array = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
kernel = np.array([1, -1, 0])
conv = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# Output array
for ii in range(8):
    conv[ii] = (kernel * array[ii:ii+3]).sum()

# Print conv
print(conv)

# output: [ 1 -1  1 -1  1 -1  1 -1  0  0]




##                  Image convolutions                  ##
kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
result = np.zeros(im.shape)

# Output array
for ii in range(im.shape[0] - 3):
    for jj in range(im.shape[1] - 3):
        result[ii, jj] = (im[ii:ii+3, jj:jj+3] * kernel).sum()

# Print result
print(result)

# output:
    [[2.68104586 2.95947725 2.84313735 ... 0.         0.         0.        ]
     [3.01830077 3.07058835 3.05098051 ... 0.         0.         0.        ]
     [2.95163405 3.09934652 3.20261449 ... 0.         0.         0.        ]
     ...
     [0.         0.         0.         ... 0.         0.         0.        ]
     [0.         0.         0.         ... 0.         0.         0.        ]
     [0.         0.         0.         ... 0.         0.         0.        ]]





##                  Defining image convolution kernels                  ##
# Vertical line in images:
np.array([[-1, 1, -1], 
          [-1, 1, -1], 
          [-1, 1, -1]])

def convolution(image, kernel):
    kernel = kernel - kernel.mean()
    result = np.zeros(image.shape)

    for ii in range(image.shape[0]-2):
        for jj in range(image.shape[1]-2):
            result[ii, jj] = np.sum(image[ii:ii+3, jj:jj+3] * kernel)

    return result


# Part 1
kernel = np.array([[-1, -1, -1], 
          [1, 1, 1], 
          [-1, -1, -1]])


# Part 2
kernel = np.array([[-1, -1, -1], 
          [-1, 1, -1], 
          [-1, -1, -1]])


# Part 3
kernel = np.array([[1, 1, 1], 
          [1, -1, 1], 
          [1, 1, 1]])
