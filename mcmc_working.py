import rebound
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
from scipy.stats import norm
from scipy import stats
print(rebound.__build__)




N = 20
times = 10.*np.sort(np.random.random(N))
rv = np.zeros((N))

sim = rebound.Simulation()
sim.add(m=1.)
sim.add(m=1e-3, a=1)
sim.move_to_com()



for i, t in enumerate(times):
    sim.integrate(t)
    rv[i] = sim.particles[0].vx + 1e-4*np.random.random()

#fig = plt.figure(figsize=(15,6))
#ax = plt.subplot(121)
#plt.plot(times,rv,"o")



def chi2(a):
    sim = rebound.Simulation()
    sim.add(m=1.)
    sim.add(m=1e-3, a=a)
    v1 = sim.add_variation()
    v2 = sim.add_variation(order=2, first_order=v1)
    v1.vary(1,"a")
    v2.vary(1,"a")
    
    sim.move_to_com()
    chi2 = 0.
    dchi2 = 0.
    ddchi2 = 0.
    for i, t in enumerate(times):
        sim.integrate(t)
        err = 1e-4
        chi2 += (sim.particles[0].vx-rv[i])**2/err**2
        dchi2 += 2.*v1.particles[0].vx*(sim.particles[0].vx-rv[i])/err**2   
        ddchi2 += 2.*v2.particles[0].vx*(sim.particles[0].vx-rv[i])/err**2 +2.*(v1.particles[0].vx)**2/err**2

    return -chi2, -dchi2, -ddchi2


print chi2(1.0000)



def q(ts,tn,eps):
    logP_n, dlogP_n, ddlogP_n = chi2(tn)
    Ginvn = -1./ddlogP_n
    mun = tn+eps**2*Ginvn*dlogP_n/2.
 
    return stats.multivariate_normal.logpdf(ts,mean=mun, cov=eps**2*Ginvn)


a = 1.03
chi2(1.)
Niter = 100
a_chain = np.zeros(Niter)
P_chain = np.zeros(Niter)
for i in range(Niter):
    a_chain[i] = a
    logP, dlogP, ddlogP = chi2(a)
    Ginv = -1./ddlogP
    eps = 1.
    
    a_proposal = a + eps**2*Ginv*dlogP/2.+eps*np.sqrt(Ginv)*np.random.randn()

    logP_proposal, dlogP_proposal, ddlogP_proposal = chi2(a_proposal)

    q1 = q(a,a_proposal,eps)
    q2 = q(a_proposal,a,eps)
    #print(q1,q2,q1-q2)
    if np.exp(logP_proposal-logP +q1 - q2) > (np.random.uniform()):
        a = a_proposal




fig = plt.figure(figsize=(15,6))
ax = plt.subplot(121)
plt.plot(range(Niter),a_chain,"o")
#ax = plt.subplot(122)
#plt.plot(range(Niter),P_chain,"o")
plt.show()
