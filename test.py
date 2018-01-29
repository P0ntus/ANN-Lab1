import math
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle

NUM_POINTS = 200
learning_rate = 0.0001

# Class that represents a 2d input and what type it should be classified as
class Point:
    def __init__(self, x, y, c):
        self.x = x
	self.y = y
	self.classification = c


# Generate points with multivariate normal distribution
meanA = [1, 3]
meanB = [-1, -3]
# cov = [[1, 0], [-5, 10]]  # diagonal covariance
covA = [[1, 0], [-5, 10]]  # diagonal covariance
covB = [[1, 0], [5, 10]]
points = []

Class_A = np.random.multivariate_normal(meanA, covA, NUM_POINTS/2).T
for x in range(0, NUM_POINTS/2):
    points.append(Point(Class_A[0][x], Class_A[1][x], 1))

Class_B = np.random.multivariate_normal(meanB, covB, NUM_POINTS/2).T
for x in range(0, NUM_POINTS/2):
    points.append(Point(Class_B[0][x], Class_B[1][x], -1))

shuffle(points)

# Initialize input output and target pattern based on earlier generated points
input_pattern = []
output_pattern = []
target_pattern = []
for point in points:
    input_pattern.append([point.x, point.y])
    target_pattern.append(point.classification)

input_pattern = np.transpose(input_pattern)


# Initiate weights using small random numbers drawn from the normal distribution with zero mean
weight = []
mu, sigma = 0, 0.1
weight.append(np.random.normal(mu, sigma, 1)[0])
weight.append(np.random.normal(mu, sigma, 1)[0])
weight.append(np.random.normal(mu, sigma, 1)[0])


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

def normalize( v ):
   result = 0
   for x in v :
	result = result + x*x
   result = math.sqrt( result )

   for x in range(0, len(v)):
	v[x] = v[x] / result
   return v

pat = np.append(input_pattern, [[1] * NUM_POINTS], 0)
# Delta Learning
error = []
for x in range(0, 10000):
    output_pattern = np.dot(np.transpose(weight), pat)
    for y in range(0, 200):
        if output_pattern[y] > 0:
	    output_pattern[y] = 1
	else:
	    output_pattern[y] = -1
    error = np.subtract(target_pattern, output_pattern)
    weight += np.dot(learning_rate, np.dot(error, np.transpose(pat)))
    normalize(weight)
    print(np.sum(error))



def phi( h_input ):
   h_output = np.divide(2 , ( 1 + np.exp( - h_input ) ) ) - 1
   return h_output

# Two-layer perceptron
STEP_LENGTH = 2
input_size = 2
output_size = 1
hidden_layer_size = 20

# Initiate weights using small random numbers drawn from the normal distribution with zero mean
mu, sigma = 0, 0.1

v = []
dv = []
for x in range(0, hidden_layer_size):
	v.append(np.random.normal(mu, sigma, input_size + 1))
	dv.append(np.random.normal(mu, sigma, input_size + 1))

v = np.array(v)
dv = np.array(dv)

w = []
dw = []
for x in range(0, output_size):
	w.append(np.random.normal(mu, sigma, hidden_layer_size + 1))
	dw.append(np.random.normal(mu, sigma, hidden_layer_size + 1))

w = np.array(w)
dw = np.array(dw)
alpha = 0.9
# print(v)

# print(w)
#transpose
#input_pattern = np.append(input_pattern, [[1] * NUM_POINTS], 0)
#np.delete(input_pattern, -1, 0)
#print(np.delete(input_pattern, -1, 0))

pat = np.append(input_pattern, [[1] * NUM_POINTS], 0)

# Forward pass
for x in range(0, 100000000):
	h_input = np.dot(v, np.append(input_pattern, [[1] * NUM_POINTS], 0))
	h_output = np.append(phi( h_input ), [[1] * NUM_POINTS], 0)
	o_input = np.dot(w, h_output)
	o_output = phi( o_input )

	# Backward pass
	delta_o = (o_output - np.asarray(target_pattern)) * ((1 + o_output) * (1 - o_output)) * 0.5
	delta_h = (np.dot(np.transpose(w), delta_o)) * ((1 + h_output) * (1 - h_output)) * 0.5
	delta_h = np.delete(delta_h, -1, 0)
	#print(delta_h)
	# Weight update
	
	#print(dv[0])
	#print(np.multiply(delta_h, input_pattern).shape)
	dv = (dv * alpha) - (np.dot(delta_h, np.transpose(pat))) * (1 - alpha)
	dw = (dw * alpha) - (np.dot(delta_o, np.transpose(h_output))) * (1 - alpha)
	v = v + dv * learning_rate
	w = w + dw * learning_rate

	error = np.subtract(target_pattern, o_output)
	print(np.sum(error))


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
output_pattern

# Plot separating line
y1 = - (weight[0] / weight[1]) * -5 - (weight[2] / weight[1])
y2 = - (weight[0] / weight[1]) * 5 - (weight[2] / weight[1])

plt.plot([-5, 5], [y1, y2])
plt.axis('equal')
plt.show()





