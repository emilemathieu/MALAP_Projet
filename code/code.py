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

def eta(w,g,t,i):
    print w.shape
    print g.shape
    print w[:,t].shape
    print X[i,:].shape
    a = -np.inner(w[:,t],X[i,:])
    return expit(a-g[t])
    
def f(x):
    #eta(t,i)*p[i,1]+(1-eta(t,i))**p[i,0]
    for i in range(X.shape[0]):
        pass
        #np.log(sum([p[i,0]+eta(t,i)*(p[i,1]-p[i,0]) for t in range(nbOfExperts)]))
        #np.log()
    
iter = 0
while np.inner(alpha-alphaNew,alpha-alphaNew)+(beta-betaNew)**2>espilon:
    print "iteration:",iter
    alpha = alphaNew
    beta = betaNew
    ###E-STEP
        #((1-eta(t,i))**abs(Y[i,t]-Z[i]))*(eta(t,i)**(1-abs(Y[i,t]-Z[i])))
    for i,x in enumerate(X):
        #P0 = ((1-eta(t,i))**abs(Y[i,t]-0))*(eta(t,i)**(1-abs(Y[i,t]-0)))
        #P1 = ((1-eta(t,i))**abs(Y[i,t]-1))*(eta(t,i)**(1-abs(Y[i,t]-1)))
        P2 = expit(-np.inner(alpha,X[i,:])-beta)
        p[i,1]=np.prod([((1-eta(weight,gamma,t,i))**abs(Y[i,t]-1))*(eta(gamma,beta,t,i)**(1-abs(Y[i,t]-1)))*P2 for t in range(nbOfExperts)])
        p[i,0]=np.prod([((1-eta(weight,gamma,t,i))**abs(Y[i,t]-0))*(eta(gamma,beta,t,i)**(1-abs(Y[i,t]-0)))*(1-P2) for t in range(nbOfExperts)])
        
    
    ###M-STEP
    #optimize.fmin_bfgs(f=,x0=,fprime=)
    
    iter = iter + 1