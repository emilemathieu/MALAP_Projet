# -*- coding: utf-8 -*-
"""
Created on Tue May 19 11:12:00 2015

@author: EmileMathieu
"""

import numpy as np # module pour les outils mathématiques
import matplotlib.pyplot as plt # module pour les outils graphiques
from sklearn.cluster import KMeans #module pour trouver des clusters par la méthode k-means
from scipy import optimize #module pour optimiser des fonction
from scipy.special import expit

#cd ~/Desktop/IMI/MALAP/Projet/code/

#########Modeling annotator expertise:
#########Learning when everybody knows a bit of something

nbOfExperts = 5

### READING DATA
def load_ionosphere(filename):
    with open(filename,"r") as f:
                 f.readline()
                 data =[ [x for x in l.split(',')] for l in f]
    tmp = np.array(data)
    tmp[:,34][np.where(tmp[:,34]=='b\n')]=0
    tmp[:,34][np.where(tmp[:,34]=='g\n')]=1
    print tmp.shape
    np.random.shuffle(tmp)
    return tmp[:,0:34].astype(float),tmp[:,34].astype(int)
    
X,Z = load_ionosphere("ionosphere.txt")

###CLUSTERING DATA WITH KMEANS
kmeans = KMeans(nbOfExperts)
clusters = kmeans.fit_predict(X)

###CREATING EXPERTS LABELS
Y=np.zeros([X.shape[0],nbOfExperts])
Y[:,clusters]=Z
for i,cluster in enumerate(clusters):
    Y[i,cluster]=Z[i]
    for j in range(Y.shape[1]):
        if j!=i and np.random.random_sample()<=0.35:
            Y[i,j]=1-Z[i]
            
###EM-ALGORITHM
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
    return res
    
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
    
iter = 0
while (np.inner(alpha-alphaNew,alpha-alphaNew)+(beta-betaNew)**2>espilon and iter<3):
    print "iteration:",iter
    ###E-STEP
    print "E-STEP"
    for i,x in enumerate(X):
        #P0 = ((1-eta(t,i))**abs(Y[i,t]-0))*(eta(t,i)**(1-abs(Y[i,t]-0)))
        #P1 = ((1-eta(t,i))**abs(Y[i,t]-1))*(eta(t,i)**(1-abs(Y[i,t]-1)))
        P2 = expit(-np.inner(alpha,X[i,:])-beta)
        p[i,1]=np.prod([((1-eta(weight,gamma,t,i))**abs(Y[i,t]-1))*(eta(weight,gamma,t,i)**(1-abs(Y[i,t]-1)))*P2 for t in range(nbOfExperts)])
        p[i,0]=np.prod([((1-eta(weight,gamma,t,i))**abs(Y[i,t]-0))*(eta(weight,gamma,t,i)**(1-abs(Y[i,t]-0)))*(1-P2) for t in range(nbOfExperts)])
        #p[i,0]=1-p[i,1]
        r = p[i,1]+p[i,0]
        p[i,1]=p[i,1]/r
        p[i,0]=p[i,0]/r
    
    ###M-STEP
    print "M-STEP"
    argmax = optimize.fmin_bfgs(f=fopt,x0=np.ones(alpha.shape[0]+1+gamma.shape[0]+np.prod(weight.shape)),fprime=gradfopt)
    alpha = argmax[0:alpha.shape[0]]
    beta = argmax[alpha.shape[0]:alpha.shape[0]+1][0]
    gamma = argmax[alpha.shape[0]+1:alpha.shape[0]+1+gamma.shape[0]]
    weight = np.reshape(argmax[alpha.shape[0]+1+gamma.shape[0]:alpha.shape[0]+1+gamma.shape[0]+np.prod(weight.shape)],(weight.shape[0],weight.shape[1]))
    iter = iter + 1
    
test = np.hstack((alpha,beta,gamma,np.reshape(weight,np.prod(weight.shape))))
print f(test)
print gradf(test)