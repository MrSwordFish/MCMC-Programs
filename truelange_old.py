import numpy as np
from scipy import stats
#from scipy.stats import multivariate_normal
import random
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy
#################################
eps = 0.2
iPos = 4
ite = 800
#################################
def logProb(x):
	#return norm.pdf(x, 1, 0.5)
	return -x**2/1.0
def logProbGrad(x):
	#return (norm.pdf(x+0.01, 1, 0.5) - norm.pdf(x-0.01, 1, 0.5))/0.02
	return -2.0*x/1.0
def logProbGrad2(x):
	#return (norm.pdf(x+0.01, 1, 0.5) + norm.pdf(x-0.01, 1, 0.5) -2.0*norm.pdf(x, 1, 0.5) )/(0.01**2)
	return -2.0/1.0

def mu(t, eps, Ginv, grad):
	return t + (eps**2.0)*(np.dot(Ginv, grad)/2.0)

def transition(t,mu,eps,Ginv):
	print "_ stuff"
	print t
	print mu
	print eps
	print Ginv
	print stats.norm.logpdf(mu, mu, eps*2*Ginv)
	print "^ stuff"
	return stats.norm.logpdf(t,mu,eps*2*Ginv)
#################################
#Actual algorithm...
currPos = iPos
#X and Y for plotting later...
X = np.zeros(shape=(ite), dtype=float)
Y = np.zeros(shape=(ite), dtype=float)

for i in range(0,ite):
	#Variables of the Old
	G = -logProbGrad2(currPos)
	Ginv = 1.0/G #np.linalg.inv(G)
	GinvDecomp = Ginv**(1/2) #np.linalg.cholesky(Ginv)
	mU = mu(currPos, eps, Ginv,logProbGrad(currPos))
	propose = mU + eps*np.dot(GinvDecomp, np.random.normal(0.0,1.0,1))
	#Variables of the New
	Gp = -logProbGrad2(propose)
	Ginvp = 1.0/Gp #np.linalg.inv(Gp)
	GinvDecompp = Ginvp**(1/2) #np.linalg.cholesky(Ginvp)
	mUp = mu(propose, eps, Ginvp, logProbGrad(propose))
	#Transition Probabilities
	currTr = transition(propose,mU,eps,Ginv) #q(tStar,t)
	propTr = transition(currPos,mUp,eps,Ginvp) #q(t,tStar)
	print currTr
	print propTr
	#print "prob_new {a}, prob_curr {b}, new/current {c}, pdfsymmetry {d}".format(a=logProb(propose),b=logProb(currPos),c=logProb(propose)/logProb(currPos),d=propTr/currTr)
	if( np.log(random.random()) < (logProb(propose)+propTr)-(logProb(currPos)+currTr)):
		print "accepted"
		currPos = propose
	#Store the values for plotting.
	X[i] = i
	Y[i] = currPos

#Use X and Y to plot the path followed.
plt.plot(X, Y, 'k')
plt.axis([-1, ite+1, -1, iPos+2])
plt.show()
