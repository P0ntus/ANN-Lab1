import math
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle

NUM_POINTS = 200

# Class that represents a 2d input and what type it should be classified as
class Point:
    def __init__(self, x, y, c):
        self.x = x
	self.y = y
	self.classification = c


# Generate points with multivariate normal distribution
meanA = [3, 3]
meanB = [-3, -3]
cov = [[1, 0], [-5, 10]]  # diagonal covariance
points = []

Class_A = np.random.multivariate_normal(meanA, cov, NUM_POINTS/2).T
for x in range(0, NUM_POINTS/2):
    points.append(Point(Class_A[0][x], Class_A[1][x], 1))

Class_B = np.random.multivariate_normal(meanB, cov, NUM_POINTS/2).T
for x in range(0, NUM_POINTS/2):
    points.append(Point(Class_B[0][x], Class_B[1][x], -1))

shuffle(points)

# Initialize input output and target pattern based on earlier generated points
input_pattern = []
output_pattern = []
target_pattern = []
for point in points:
    input_pattern.append([point.x, point.y, 1])
    target_pattern.append(point.classification)

input_pattern = np.transpose(input_pattern)


# Initiate weights using small random numbers drawn from the normal distribution with zero mean
'''
weight = []
mu, sigma = 0, 0.1
weight.append(np.random.normal(mu, sigma, 1)[0])
weight.append(np.random.normal(mu, sigma, 1)[0])
weight.append(np.random.normal(mu, sigma, 1)[0])
learning_rate = 0.00001
'''

'''
# Perceptron Learning
error = []
for x in range(0, 1000):
    output_pattern = np.dot(np.transpose(weight), input_pattern)
    for y in range(0, 200):
        if output_pattern[y] > 0:
	    output_pattern[y] = 1
	else:
	    output_pattern[y] = -1
    error = np.subtract(target_pattern, output_pattern)
    weight += np.dot(learning_rate, np.dot(error, np.transpose(input_pattern)))
    #print(np.sum(error))
'''

'''
# Delta Learning
error = []
for x in range(0, 1000):
    output_pattern = np.dot(np.transpose(weight), input_pattern)
    error = np.subtract(target_pattern, output_pattern)
    weight += np.dot(learning_rate, np.dot(error, np.transpose(input_pattern)))
    print(np.sum(error))
'''


def phi( h_input ):
   h_output = np.divide(2 , ( 1 + np.exp( - h_input ) ) ) - 1
   return h_output

# Two-layer perceptron
NUM_NODES = 4
NUM_ITERATIONS = 10
STEP_LENGTH = 2
v_size = 4
w_size = 3

# Initiate weights using small random numbers drawn from the normal distribution with zero mean
mu, sigma = 0, 0.1

v = []
#print( input_pattern )
for x in range(0, v_size):
	v.append(np.random.normal(mu, sigma, NUM_POINTS))

w = []
for x in range(0, w_size):
	w.append(np.random.normal(mu, sigma, v_size))

# print(v)

# print(w)
#print (input_pattern)
#print (np.transpose(input_pattern))
# Forward pass
h_input = np.dot(v, np.transpose(input_pattern))
print(h_input)
h_output = phi( h_input )
print(h_output)

o_input = np.dot(w, h_output)
o_output = phi( o_input )
#print(o_output)

delta_o = (o_output - np.asarray(target_pattern)) * ((1 + o_output) * (1 - o_output)) * 0.5
#print(delta_o)


'''
# Plot the points
positive = []
negative = []
for x in range(0, NUM_POINTS):
    if target_pattern[x] == 1:
	positive.append([input_pattern[0][x], input_pattern[1][x]])
    else:
	negative.append([input_pattern[0][x], input_pattern[1][x]])

positive = np.transpose(positive)
negative = np.transpose(negative)

plt.plot(positive[0], positive[1], 'x')
plt.plot(negative[0], negative[1], 'o')


# Plot separating line
y1 = - (weight[0] / weight[1]) * -15 - (weight[2] / weight[1])
y2 = - (weight[0] / weight[1]) * 15 - (weight[2] / weight[1])

plt.plot([-15, 15], [y1, y2])
plt.axis('equal')
plt.show()
'''



