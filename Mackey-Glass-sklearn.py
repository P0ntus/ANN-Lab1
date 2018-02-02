import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from random import shuffle

from sklearn.neural_network import MLPRegressor

np.set_printoptions(threshold=np.nan) #Always print the whole matrix

POINTS_TO_GEN = 1500
OFFSET = 300
NUM_POINTS = POINTS_TO_GEN - OFFSET
learning_rate = 0.0001

TRAINING_SIZE = 1000
TEST_SIZE = NUM_POINTS - TRAINING_SIZE

def MGE(t, x):
	return x[t] + ((0.2 * x[t-25]) / (1 + x[t-25]**10)) - 0.1 * x[t]

prev_x = []
# Add 25 zero values so that prev_x[t] will return 0 if t is lower than 0
for t in range(0, 25):
	prev_x.append(0)

x = 1.5
# Calculate x values
prev_x.append(x)
for t in range(25, POINTS_TO_GEN + 50):
	prev_x.append(MGE(t, prev_x))

# Calculate input values for the ANN
input_pattern = []
target_pattern = []
for t in range(OFFSET+50, POINTS_TO_GEN + 50):
	input_pattern.append([prev_x[t-25], prev_x[t-20], prev_x[t-15], prev_x[t-10], prev_x[t-5]])
	target_pattern.append(prev_x[t])

# Split into training and test set
training_target_pattern = target_pattern[0:TRAINING_SIZE]
test_target_pattern = target_pattern[TRAINING_SIZE:NUM_POINTS]

training_input_pattern = input_pattern[0:TRAINING_SIZE]
test_input_pattern = input_pattern[TRAINING_SIZE:NUM_POINTS]

input_pattern = np.array(input_pattern)
#print(len(target_pattern))
#input_pattern = (input_pattern - input_pattern.mean(axis=0))/input_pattern.var(axis=0)
input_pattern = np.transpose(input_pattern)
target_pattern = np.array(target_pattern)
target_pattern = (target_pattern - target_pattern.mean(axis=0))/target_pattern.std(axis=0) / 2.5

#for t in range(0, 5):
	#input_pattern[t] = normalize(input_pattern[t])

#target_pattern = normalize(target_pattern)

input_size = 5
output_size = 1
hidden_layer_size = 15

reg = MLPRegressor(hidden_layer_sizes=hidden_layer_size)
reg = reg.fit(np.transpose(input_pattern), target_pattern)
output = reg.predict(np.transpose(input_pattern))

'''
plt.plot(output)
plt.plot(target_pattern)
plt.legend(['y = ANN', 'y = TARGET'])
plt.ylabel('some numbers')
plt.show()
'''


