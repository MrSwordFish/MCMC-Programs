###############################################################################################################
#imports
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import random
import math
###############################################################################################################
#User defined stuff
#Here we have the potential and it's gradient
def	Ufunc(x):
	return norm.pdf(x, 0, 1)
#This is the step size
eps = 0.2
#This is the number of MC iterations
N = 800
#
variance = 0.08
#Starting vector
q = 3
###############################################################################################################
#Functions
def metroMC(variance, curr):
	x = curr
	x_t = np.random.normal(curr,variance)
	a_1 = Ufunc(x_t)/Ufunc(x)
	a_2 = norm.pdf(x,x_t,variance)/ norm.pdf(x_t,x,variance)
	print a_1*a_2
	if( random.random() < a_1*a_2 ):
		return x_t
	else:
		return x
###############################################################################################################
print "Start"
history = np.zeros(1000)
for j in range (0,1000):
	innerY = np.zeros(N)
	innerX = np.zeros(N)
	X = q
	for i in range (0,N):
		X = metroMC(variance, X)
		innerY[i] = X
		innerX[i] = i
	plt.plot(innerX, innerY, 'k')
	plt.plot(innerX, 4*np.ones(len(innerX)), 'r')
	plt.show()
	print "The result is : {one}".format(one=X)
	history[j] = X
print history
fig = plt.figure()
plt.hist(history, bins = 16)
plt.show()
print "End"