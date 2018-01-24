import numpy as np
import matplotlib.pyplot as plt
from random import shuffle

class MyClass:
    def __init__(self, x, y, c):
        self.x = x
	self.y = y
	self.classification = c

meanA = [3, 3]
meanB = [-3, -3]
cov = [[1, 0], [-5, 10]]  # diagonal covariance
points = []

Class_A = np.random.multivariate_normal(meanA, cov, 100).T
for x in range(0, 100):
    points.append(MyClass(Class_A[0][x], Class_A[1][x], 1))

Class_B = np.random.multivariate_normal(meanB, cov, 100).T
for x in range(0, 100):
    points.append(MyClass(Class_B[0][x], Class_B[1][x], -1))

shuffle(points)


input_pattern = []
output_pattern = []
target_pattern = []

for point in points:
    input_pattern.append([point.x, point.y, 1])
    target_pattern.append(point.classification)

input_pattern = np.transpose(input_pattern)

#for x in range(0, 200):
 #   print(input_pattern[x][0], input_pattern[x][1], input_pattern[x][2], target_pattern[x])

positive = []
negative = []
for x in range(0, 200):
    if target_pattern[x] == 1:
	positive.append([input_pattern[0][x], input_pattern[1][x]])
    else:
	negative.append([input_pattern[0][x], input_pattern[1][x]])

positive = np.transpose(positive)
negative = np.transpose(negative)

plt.plot(positive[0], positive[1], 'x')
plt.plot(negative[0], negative[1], 'o')
plt.axis('equal')
plt.show()

weight = []
mu, sigma = 0, 0.1
weight.append(np.random.normal(mu, sigma, 1))
weight.append(np.random.normal(mu, sigma, 1))
weight.append(np.random.normal(mu, sigma, 1))




