# -*- coding: utf-8 -*-
"""
Created on Tue May 19 11:12:00 2015

@author: EmileMathieu
"""

import numpy as np # module pour les outils mathématiques
import matplotlib.pyplot as plt # module pour les outils graphiques
from sklearn.cluster import KMeans #module pour trouver des clusters par la méthode k-means
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize #module pour optimiser des fonction
from scipy.special import expit
from sklearn import cross_validation as cv
from collections import Counter


#cd ~/Desktop/IMI/MALAP/Projet/code/

#########Modeling annotator expertise:
#########Learning when everybody knows a bit of something

#################   Data Loading and Formating    ################# 

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
print kmeans.inertia_

###CREATING EXPERTS LABELS
Y=np.zeros([X.shape[0],nbOfExperts])
Y[:,clusters]=Z
for i,cluster in enumerate(clusters):
    Y[i,cluster]=Z[i]
    for j in range(Y.shape[1]):
        if j!=i and np.random.random_sample()<=0.35:
            Y[i,j]=1-Z[i]
            
#################    Learning parameters    ################# 
            
### Initialization            
alpha = np.ones(X.shape[1])
beta = 1
espilon = 1e-5
p = np.zeros([Z.shape[0],2])

alphaNew = np.zeros(X.shape[1])
betaNew = 0
weight = np.zeros([X.shape[1],nbOfExperts])
gamma = np.zeros(nbOfExperts)

def eta(U,V,t,i):
    return expit(np.inner(U[:,t],X[i,:])+V[t])
    
### To be optimized Function    
def fopt(x):
    a = x[0:alpha.shape[0]]
    b = x[alpha.shape[0]:alpha.shape[0]+1][0]
    g = x[alpha.shape[0]+1:alpha.shape[0]+1+gamma.shape[0]]
    w = np.reshape(x[alpha.shape[0]+1+gamma.shape[0]:alpha.shape[0]+1+gamma.shape[0]+np.prod(weight.shape)],(weight.shape[0],weight.shape[1]))
    res = 0
    for i in range(X.shape[0]):
        k = sum([p[i,1]*((1-Y[i,t])*np.log(1-eta(w,g,t,i))+Y[i,t]*np.log(eta(w,g,t,i)))+p[i,0]*(Y[i,t]*np.log(1-eta(w,g,t,i))+(1-Y[i,t])*np.log(eta(w,g,t,i))) for t in range(nbOfExperts)])
        l = p[i,1]*np.log(expit(np.inner(a,X[i,:])+b)) + p[i,0]*np.log(1-expit(np.inner(a,X[i,:])+b))
        res = res + k + l
    return -res
    
### Gradient of to be optimized function    
def gradfopt_Personal(x):
    a = x[0:alpha.shape[0]]
    b = x[alpha.shape[0]:alpha.shape[0]+1][0]
    g = x[alpha.shape[0]+1:alpha.shape[0]+1+gamma.shape[0]]
    w = np.reshape(x[alpha.shape[0]+1+gamma.shape[0]:alpha.shape[0]+1+gamma.shape[0]+np.prod(weight.shape)],(weight.shape[0],weight.shape[1]))
    
    gradalpha = np.zeros(alpha.shape[0])  
    gradbeta = 0
    gradgamma = 0
    gradweight = np.zeros(np.prod(weight.shape))  
    
    for i,Xi in enumerate(X):
        gradalpha = gradalpha + (p[i,1]-expit(np.inner(a,x)+b))*Xi
        gradbeta = gradbeta + p[i,1]-expit(np.inner(a,Xi)+b)
        gradgamma = gradgamma + p[i,1]*(-eta(w,g,t,i)+Y[i,t]) + p[i,0]*(1-eta(w,g,t,i)-Y[i,t])
        gradweight = gradweight + (p[i,1]*(-eta(w,g,t,i)+Y[i,t]) + p[i,0]*(1-eta(w,g,t,i)-Y[i,t]))*Xi
    return -np.hstack((gradalpha,gradbeta,gradgamma,gradweight))
    
def gradfopt(x):
    a = x[0:alpha.shape[0]]
    b = x[alpha.shape[0]:alpha.shape[0]+1][0]
    g = x[alpha.shape[0]+1:alpha.shape[0]+1+gamma.shape[0]]
    w = np.reshape(x[alpha.shape[0]+1+gamma.shape[0]:alpha.shape[0]+1+gamma.shape[0]+np.prod(weight.shape)],(weight.shape[0],weight.shape[1]))
    
    gradalpha = np.zeros(alpha.shape[0])  
    gradbeta = 0
    gradgamma = np.zeros(gamma.shape[0])  
    gradweight = np.zeros(np.prod(weight.shape))  
    
    for i,Xi in enumerate(X):
        gradalpha = gradalpha + (p[i,1]-p[i,0])*expit(np.inner(a,Xi)+b)*(1-expit(np.inner(a,Xi)+b))*Xi
        gradbeta = gradbeta + (p[i,1]-p[i,0])*expit(np.inner(a,Xi)+b)*(1-expit(np.inner(a,Xi)+b))
        gradgamma = gradgamma + [(-1)**Y[i,t]*(p[i,0]-p[i,1])*expit(np.inner(w[:,t],Xi)+g[t])*(1-expit(np.inner(w[:,t],Xi)+g[t]))  for t in range(nbOfExperts)]
        gradweight = gradweight + np.reshape(np.array([(-1)**Y[i,t]*(p[i,0]-p[i,1])*expit(np.inner(w[:,t],Xi)+g[t])*(1-expit(np.inner(w[:,t],Xi)+g[t]))*Xi  for t in range(nbOfExperts)]),np.prod(weight.shape))
    return -np.hstack((gradalpha,gradbeta,gradgamma,gradweight))

    
####EM-Algorithm
iter = 0
while (np.inner(alpha-alphaNew,alpha-alphaNew)/alpha.shape[0]+(beta-betaNew)**2>espilon):
    print "iteration:",iter
    print "convergence criterion:", np.inner(alpha-alphaNew,alpha-alphaNew)+(beta-betaNew)**2
    
    if iter==0:
        theta_old=0.1*np.ones(alpha.shape[0]+1+gamma.shape[0]+np.prod(weight.shape))
    else:
        theta_old=theta_new
        
    print "f(theta)=", fopt(theta_old)    
    
    ###E-STEP
    alpha = theta_old[0:alpha.shape[0]]
    beta = theta_old[alpha.shape[0]:alpha.shape[0]+1][0]
    gamma = theta_old[alpha.shape[0]+1:alpha.shape[0]+1+gamma.shape[0]]
    weight = np.reshape(theta_old[alpha.shape[0]+1+gamma.shape[0]:alpha.shape[0]+1+gamma.shape[0]+np.prod(weight.shape)],(weight.shape[0],weight.shape[1]))
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
    
    ###M-STEP
    print "M-STEP"
    r#es = minimize(fopt, theta_old, method='L-BFGS-B', options={'disp': True})
    res = minimize(fopt, theta_old, method='L-BFGS-B', options={'disp': True, 'gtol':1})
    #res = minimize(fopt, theta_old, options={'disp': True, 'maxiter': 5})
    #res = minimize(fopt, theta_old, method='BFGS', options={'disp': True, 'maxiter': 10})
    
    #res = minimize(fopt, theta_old, method='L-BFGS-B',jac=gradfopt, options={'disp': True})
    
    theta_new = res.x
    alphaNew = theta_new[0:alpha.shape[0]]
    betaNew = theta_new[alpha.shape[0]:alpha.shape[0]+1][0]
    gamma = theta_new[alpha.shape[0]+1:alpha.shape[0]+1+gamma.shape[0]]
    weight = np.reshape(theta_new[alpha.shape[0]+1+gamma.shape[0]:alpha.shape[0]+1+gamma.shape[0]+np.prod(weight.shape)],(weight.shape[0],weight.shape[1]))
    iter = iter + 1

print "convergence criterion:", np.inner(alpha-alphaNew,alpha-alphaNew)+(beta-betaNew)**2
print "f(theta_new)=",fopt(theta_new)
alpha = theta_new[0:alpha.shape[0]]
beta = theta_new[alpha.shape[0]:alpha.shape[0]+1][0]
gamma = theta_new[alpha.shape[0]+1:alpha.shape[0]+1+gamma.shape[0]]
weight = np.reshape(theta_new[alpha.shape[0]+1+gamma.shape[0]:alpha.shape[0]+1+gamma.shape[0]+np.prod(weight.shape)],(weight.shape[0],weight.shape[1]))

################ Classification ################# 


### ROC for EM-algorithm
sensitivity=np.zeros(9)
antispecificity=np.zeros(9)
for s in range(1,10):
    threshold=s*1.0/10
    print threshold
    for i,instance in enumerate(X):
        sensitivity[s-1]=sensitivity[s-1]+int(expit(np.inner(alpha,instance)+beta)>threshold)*int(Z[i]==1)
        antispecificity[s-1]=antispecificity[s-1]+int(expit(np.inner(alpha,instance)+beta)<threshold)*int(Z[i]==0)
        #score=score+int(Z[i]==int(expit(np.inner(alphaNew,instance)+betaNew)>threshold))
    sensitivity[s-1]=sensitivity[s-1]*1.0/sum(Z==1)
    antispecificity[s-1]=1-(antispecificity[s-1]*1.0/sum(Z==0))
    print "sensitivity", sensitivity[s-1]
    print "antispecificity", antispecificity[s-1]

plt.plot(antispecificity,sensitivity)
    
### Classification for Logistic regression on majority voting

# Majority label
Y_maj=np.zeros_like(Z)
for i,y in enumerate(Y):
    Y_maj[i]=Counter(y).most_common()[0][0]

Majority_Logistic = LogisticRegression(verbose=True)
kf= cv.KFold(X.shape[0],n_folds=5)
res_test=[]
for cvtrain,cvtest in kf:
    Majority_Logistic.fit(X[cvtrain],Y_maj[cvtrain])
    res_test+=[Majority_Logistic.score(X[cvtest],Y_maj[cvtest])]

    