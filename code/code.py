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
from sklearn.metrics import roc_curve, auc
from scipy import interp

#cd ~/Desktop/IMI/MALAP/Projet/code/

#########Modeling annotator expertise:
#########Learning when everybody knows a bit of something
            
#################    Learning parameters    ################# 

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
    #print tmp.shape
    np.random.shuffle(tmp)
    return tmp[:,0:34].astype(float),tmp[:,34].astype(int)
    
X,Z = load_ionosphere("ionosphere.csv")

###CLUSTERING DATA WITH KMEANS
kmeans = KMeans(nbOfExperts)
clusters = kmeans.fit_predict(X)
print kmeans.inertia_

###CREATING EXPERTS LABELS
Y=np.zeros([X.shape[0],nbOfExperts])
for i,cluster in enumerate(clusters):
    Y[i,:]=Z[i]
    for j in range(Y.shape[1]):
        if j!=cluster and np.random.random_sample()<0.35:
            Y[i,j]=1-Z[i]

### Initialization            
alpha = np.ones(X.shape[1])
beta = 1
epsilon = 1e-8
p = np.zeros([Z.shape[0],2])

alphaNew = np.zeros(X.shape[1])
betaNew = 0
weight = np.zeros([X.shape[1],nbOfExperts])
gamma = np.zeros(nbOfExperts)
    
####EM-Algorithm
iter = 0
while (np.inner(alpha-alphaNew,alpha-alphaNew)/alpha.shape[0]+(beta-betaNew)**2>epsilon):
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
    #res = minimize(fopt, theta_old, method='L-BFGS-B', options={'disp': True})
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

fpr, tpr, thresholds = roc_curve(Z, expit(np.sum(np.multiply(X,np.tile(alpha,(X.shape[0],1))),1)+beta))
fpr, tpr, thresholds = roc_curve(Z, p[:,1])

roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, 'k--',label='Bernouilli ROC (area = %0.2f)' % roc_auc, lw=2)
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
    
### Classification for Logistic regression on majority voting & concatenating

def LogisticRegressionROC(Data,Label,legend):
    Majority_Logistic = LogisticRegression()
    Majority_Logistic.verbose=0
    N_folds=5
    kf= cv.KFold(X.shape[0],n_folds=N_folds)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    for i, (train, test) in enumerate(kf):
        probas_ = Majority_Logistic.fit(Data[train], Label[train]).predict_proba(Data[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(Label[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        #roc_auc = auc(fpr, tpr)
        #plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

    #plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    
    mean_tpr /= len(kf)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',color=(np.random.random_sample(), np.random.random_sample(), np.random.random_sample()),label='%s ROC (AUC: %0.2f)' % (legend,mean_auc), lw=2)
    
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('R.O.C. Curves for iono data')
    plt.legend(loc="lower right")
    #plt.show()
    
# Majority label
Y_maj=np.zeros_like(Z)
for i,y in enumerate(Y):
    Y_maj[i]=Counter(y).most_common()[0][0]

# concatenate labels
X_concatenate=np.repeat(X,5,axis=0)
Y_concatenate=np.reshape(Y,np.prod(Y.shape))

LogisticRegressionROC(X_concatenate,Y_concatenate,"L.R. Concatenation")

LogisticRegressionROC(X,Y_maj,"L.R. Majority")

LogisticRegressionROC(X,Y[:,0],"L.R.-Annotator1")
LogisticRegressionROC(X,Y[:,1],"L.R.-Annotator2")
LogisticRegressionROC(X,Y[:,2],"L.R.-Annotator3")
LogisticRegressionROC(X,Y[:,3],"L.R.-Annotator4")
LogisticRegressionROC(X,Y[:,4],"L.R.-Annotator5")
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
plt.show()
