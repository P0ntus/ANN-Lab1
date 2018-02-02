import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from random import shuffle

np.set_printoptions(threshold=np.nan) #Always print the whole matrix

learning_rate = 0.001

# Function approximation

# Generate function data
x = np.arange(-5, 5.5, 0.5)[np.newaxis].T
y = np.arange(-5, 5.5, 0.5)[np.newaxis].T 
z = np.dot(np.exp(-x*np.dot(x, 0.1)), np.transpose(np.exp(-y*np.dot(y,0.1)))) - 0.5

target_pattern = np.reshape(z, (len(z)*len(z), 1))
NUM_POINTS = len(target_pattern)
target_pattern = np.transpose(target_pattern)
xx, yy = np.meshgrid(x, y)

input_pattern = np.vstack(((np.reshape(xx, len(xx)*len(xx))), np.reshape(yy, len(yy)*len(yy))))

# Two-layer perceptron
def phi( h_input ):
   h_output = np.divide(2 , ( 1 + np.exp( - h_input ) ) ) - 1
   return h_output

input_size = 2
output_size = 1
hidden_layer_size = 25

# Initiate weights using small random numbers drawn from the normal distribution with zero mean
mu, sigma = 0, 0.1

v = []
dv = []
for i in range(0, hidden_layer_size):
	v.append(np.random.normal(mu, sigma, input_size + 1))
	dv.append(np.random.normal(mu, sigma, input_size + 1))

v = np.array(v)
dv = np.array(dv)

w = []
dw = []
for i in range(0, output_size):
	w.append(np.random.normal(mu, sigma, hidden_layer_size + 1))
	dw.append(np.random.normal(mu, sigma, hidden_layer_size + 1))

w = np.array(w)
dw = np.array(dw)
alpha = 0.9

pat = np.append(input_pattern, [[1] * NUM_POINTS], 0)

o_output = []
for i in range(0, 10000):
	# Forward pass
	h_input = np.dot(v, np.append(input_pattern, [[1] * NUM_POINTS], 0))
	h_output = np.append(phi( h_input ), [[1] * NUM_POINTS], 0)
	o_input = np.dot(w, h_output)
	o_output = phi( o_input )

	# Backward pass
	delta_o = (o_output - np.asarray(target_pattern)) * ((1 + o_output) * (1 - o_output)) * 0.5
	delta_h = (np.dot(np.transpose(w), delta_o)) * ((1 + h_output) * (1 - h_output)) * 0.5
	delta_h = np.delete(delta_h, -1, 0)
	
	# Weight update
	dv = (dv * alpha) - (np.dot(delta_h, np.transpose(pat))) * (1 - alpha)
	dw = (dw * alpha) - (np.dot(delta_o, np.transpose(h_output))) * (1 - alpha)
	v = v + dv * learning_rate
	w = w + dw * learning_rate

	error = np.subtract(target_pattern, o_output)
	print(np.sum(error))

zz = np.reshape(o_output, (len(x), len(y)))

# Plot generated values
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(xx, yy, zz, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()


'''
# Plot generated values
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(xx, yy, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
'''



