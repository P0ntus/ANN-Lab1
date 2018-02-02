import matplotlib.pyplot as plt

def MGE(t, x):
	return x[t] + ((0.2 * x[t-25]) / (1 + x[t-25]**10)) - 0.1 * x[t]


prev_x = []
for t in range(0, 25):
	prev_x.append(0)

x = 1.5
prev_x.append(x)
for t in range(25, 1000):
	prev_x.append(MGE(t, prev_x))

#plt.plot(prev_x)
#plt.ylabel('some numbers')
#plt.show()
