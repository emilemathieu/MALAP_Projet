# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 09:58:00 2015

@author: EmileMathieu
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

def rosen(x):
    """The Rosenbrock function"""
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)
    
def rosen_der(x):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
    der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
    der[-1] = 200*(x[-1]-x[-2]**2)
    return der
    
x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
res = minimize(rosen, x0, method='BFGS', jac=rosen_der, options={'disp': True})



nbOfExperts = 5

alpha = np.zeros(X.shape[1])
beta = 0
espilon = 1e-5
p = np.zeros([Z.shape[0],2])

alphaNew = np.ones(X.shape[1])
betaNew = 1
weight = np.zeros([X.shape[1],nbOfExperts])
gamma = np.zeros(nbOfExperts)

def eta(U,V,t,i):
    return expit(np.inner(U[:,t],X[i,:])+V[t])

def fopt(x):
    a = x[0:alpha.shape[0]]
    #print "a.shape", a.shape
    b = x[alpha.shape[0]:alpha.shape[0]+1][0]
    #print "b",b
    g = x[alpha.shape[0]+1:alpha.shape[0]+1+gamma.shape[0]]
    #print "g.shape", g.shape
    w = np.reshape(x[alpha.shape[0]+1+gamma.shape[0]:alpha.shape[0]+1+gamma.shape[0]+np.prod(weight.shape)],(weight.shape[0],weight.shape[1]))
    #print "w.shape", w.shape    
    res = 0
    for i in range(X.shape[0]):
        k = sum([p[i,1]*((1-Y[i,t])*np.log(1-eta(w,g,t,i))+Y[i,t]*np.log(eta(w,g,t,i)))+p[i,0]*(Y[i,t]*np.log(1-eta(w,g,t,i))+(1-Y[i,t])*np.log(eta(w,g,t,i))) for t in range(nbOfExperts)])
        l = p[i,1]*np.log(expit(np.inner(a,X[i,:])+b)) + p[i,0]*np.log(1-expit(np.inner(a,X[i,:])+b))
        res = res + k + l
        #print "k",k
        #print"l",l
        #print "res",res
    return -res
    
def gradfopt(x):
    a = x[0:alpha.shape[0]]
    b = x[alpha.shape[0]:alpha.shape[0]+1][0]
    g = x[alpha.shape[0]+1:alpha.shape[0]+1+gamma.shape[0]]
    w = np.reshape(x[alpha.shape[0]+1+gamma.shape[0]:alpha.shape[0]+1+gamma.shape[0]+np.prod(weight.shape)],(weight.shape[0],weight.shape[1]))
    
    gradalpha = np.zeros(alpha.shape[0])  
    gradbeta = 0
    gradgamma = 0
    gradweight = np.zeros(weight.shape[0])  
    
    for i in range(X.shape[0]):
        pass
        gradalpha = gradalpha + (p[i,1]-expit(np.inner(a,X[i,:])+b))*X[i,:]
        gradbeta = gradbeta + p[i,1]-expit(np.inner(a,X[i,:])+b)
        gradgamma = gradgamma + p[i,1]*(-eta(w,g,t,i)+Y[i,t]) + p[i,0]*(1-eta(w,g,t,i)-Y[i,t])
        gradweight = gradweight + (p[i,1]*(-eta(w,g,t,i)+Y[i,t]) + p[i,0]*(1-eta(w,g,t,i)-Y[i,t]))*X[i,:]
    return np.hstack((gradalpha,gradbeta,gradgamma,gradweight))
    
print "E-STEP"
for i,x in enumerate(X):
    P2 = expit(-np.inner(alpha,X[i,:])-beta)
    p[i,1]=np.prod([((1-eta(weight,gamma,t,i))**abs(Y[i,t]-1))*(eta(weight,gamma,t,i)**(1-abs(Y[i,t]-1))) for t in range(nbOfExperts)])
    p[i,1] = p[i,1]*P2
    p[i,0]=np.prod([((1-eta(weight,gamma,t,i))**abs(Y[i,t]-0))*(eta(weight,gamma,t,i)**(1-abs(Y[i,t]-0))) for t in range(nbOfExperts)])
    p[i,0] = p[i,0]*(1-P2)
    r = p[i,1]+p[i,0]
    p[i,1]=p[i,1]/r
    p[i,0]=p[i,0]/r
        
x0=np.ones(alpha.shape[0]+1+gamma.shape[0]+np.prod(weight.shape))
x1=0.1*np.ones(alpha.shape[0]+1+gamma.shape[0]+np.prod(weight.shape))
x2=0.0*np.ones(alpha.shape[0]+1+gamma.shape[0]+np.prod(weight.shape))
x3=-1*np.ones(alpha.shape[0]+1+gamma.shape[0]+np.prod(weight.shape))
res = minimize(fopt, x1, method='L-BFGS-B', options={'disp': True})