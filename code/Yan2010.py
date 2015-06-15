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
    
def load_housing(filename):
    with open(filename,"r") as f:
                 f.readline()
                 data =[ [x for x in l.split(' ') if x!=''] for l in f]
    tmp = np.array(data)
    tmp[:,13]=[(price>np.median(tmp[:,13].astype(float))).astype(int) for price in tmp[:,13].astype(float)]
    print tmp.shape
    np.random.shuffle(tmp)
    return tmp[:,0:13].astype(float),tmp[:,13].astype(float)
    
def load_glass(filename):
    with open(filename,"r") as f:
                 f.readline()
                 data =[ [x for x in l.split(',')] for l in f]
    tmp = np.array(data)
    tmp[:,10]=[(label>2).astype(int) for label in tmp[:,10].astype(int)]
    print tmp.shape
    np.random.shuffle(tmp)
    return tmp[:,1:10].astype(float),tmp[:,10].astype(int)    
    
#X,Z = load_ionosphere("../DataSets/ionosphere.csv")
X,Z = load_housing("../DataSets/housing.csv")
#X,Z = load_glass("../DataSets/glass.csv")

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
epsilon = 1e-30

alphaNew = np.zeros(X.shape[1])
betaNew = 0
weight = np.zeros([X.shape[1],nbOfExperts])
gamma = np.zeros(nbOfExperts)     
p=0.5*np.ones([X.shape[0],2])       
            
            
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
    for i,Xi in enumerate(X):
        k = sum([p[i,1]*((1-Y[i,t])*np.log(1-expit(np.inner(Xi,w[:,t])+g[t]))+Y[i,t]*np.log(expit(np.inner(Xi,w[:,t])+g[t])))+p[i,0]*(Y[i,t]*np.log(1-expit(np.inner(Xi,w[:,t])+g[t]))+(1-Y[i,t])*np.log(expit(np.inner(Xi,w[:,t])+g[t]))) for t in range(nbOfExperts)])
        l = nbOfExperts*(p[i,1]*np.log(expit(np.inner(a,Xi)+b)) + p[i,0]*np.log(1-expit(np.inner(a,Xi)+b)))
        res += k + l
    return -res
    
### Gradient of to be optimized function    
def gradfopt_EM(x):
    a = x[0:alpha.shape[0]]
    b = x[alpha.shape[0]:alpha.shape[0]+1][0]
    g = x[alpha.shape[0]+1:alpha.shape[0]+1+gamma.shape[0]]
    w = np.reshape(x[alpha.shape[0]+1+gamma.shape[0]:alpha.shape[0]+1+gamma.shape[0]+np.prod(weight.shape)],(weight.shape[0],weight.shape[1]))
    
    gradalpha = np.zeros_like(alpha)
    gradbeta = 0
    gradgamma = np.zeros_like(gamma)  
    gradweight = np.zeros(np.prod(weight.shape))  
    
    for i,Xi in enumerate(X):
        temp1=nbOfExperts*(p[i,1]-expit(np.inner(a,Xi)+b))
        gradalpha+=temp1*Xi
        gradbeta +=temp1
        #gradalpha += nbOfExperts*(p[i,1]-expit(np.inner(a,Xi)+b))*Xi
        #gradbeta +=  nbOfExperts*(p[i,1]-expit(np.inner(a,Xi)+b))
        gradgamma +=  [(1-expit(np.inner(w[:,t],Xi)+g[t]))+Y[i,t]*(p[i,1]-p[i,0])-p[i,1] for t in range(nbOfExperts)]
        gradweight +=  np.reshape(np.array([((1-expit(np.inner(w[:,t],Xi)+g[t]))+Y[i,t]*(p[i,1]-p[i,0])-p[i,1])*Xi for t in range(nbOfExperts)]),np.prod(weight.shape))
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
theta_old=0.001*np.ones(alpha.shape[0]+1+gamma.shape[0]+np.prod(weight.shape))
theta_new=0.0001*np.ones(alpha.shape[0]+1+gamma.shape[0]+np.prod(weight.shape))
#while (np.inner(alpha-alphaNew,alpha-alphaNew)+(beta-betaNew)**2>epsilon):
while((fopt(theta_old)-fopt(theta_new))**2>epsilon and np.inner(alpha-alphaNew,alpha-alphaNew)+(beta-betaNew)**2>epsilon):
    print "iteration:",iter
    print "convergence criterion:", np.inner(alpha-alphaNew,alpha-alphaNew)+(beta-betaNew)**2
    
    if iter==0:
        theta_old=0.0001*np.ones(alpha.shape[0]+1+gamma.shape[0]+np.prod(weight.shape))
    else:
        theta_old=theta_new
        
    print "f(theta)=", fopt(theta_old)    
    
    ###E-STEP
    alpha = theta_old[0:alpha.shape[0]]
    beta = theta_old[alpha.shape[0]:alpha.shape[0]+1][0]
    gamma = theta_old[alpha.shape[0]+1:alpha.shape[0]+1+gamma.shape[0]]
    weight = np.reshape(theta_old[alpha.shape[0]+1+gamma.shape[0]:alpha.shape[0]+1+gamma.shape[0]+np.prod(weight.shape)],(weight.shape[0],weight.shape[1]))
    #print "E-STEP"
    if iter==0:
        for i,Yi in enumerate(Y):
            p[i,1]=np.mean(Yi)
        p[:,0]=1-p[:,1]
    else:
        for i,Xi in enumerate(X):
            P2 = expit(np.inner(alpha,Xi)+beta)
            p[i,1]=np.prod([((1-expit(np.inner(Xi,weight[:,t])+gamma[t]))**abs(Y[i,t]-1))*(expit(np.inner(Xi,weight[:,t])+gamma[t])**(1-abs(Y[i,t]-1))) for t in range(nbOfExperts)])
            p[i,1] = p[i,1]*P2
            p[i,0]=np.prod([((1-expit(np.inner(Xi,weight[:,t])+gamma[t]))**abs(Y[i,t]-0))*(expit(np.inner(Xi,weight[:,t])+gamma[t])**(1-abs(Y[i,t]-0))) for t in range(nbOfExperts)])
            p[i,0] = p[i,0]*(1-P2)
            r = p[i,1]+p[i,0]
            p[i,1]=p[i,1]/r
            p[i,0]=p[i,0]/r
    
    ###M-STEP
    #print "M-STEP"
    #res = minimize(fopt, theta_old, method='L-BFGS-B', options={'disp': True})
    #res = minimize(fopt, theta_old, method='L-BFGS-B', options={'disp': True, 'gtol':1})
    #res = minimize(fopt, theta_old, options={'disp': True, 'maxiter': 5})
    #res = minimize(fopt, theta_old, method='BFGS', options={'disp': True, 'maxiter': 10})
    
    #res = minimize(fopt, theta_old,jac=gradfopt_EM, options={'disp': True})
    #res = minimize(fopt, theta_old, method='L-BFGS-B',jac=gradfopt_EM, options={'disp': True})
    res = minimize(fopt, theta_old,jac=gradfopt_EM , options={'disp': True})
    #res = minimize(fopt, theta_old, method='BFGS',jac=gradfopt_EM) 
    
    theta_new = res.x
    #print "f(theta°old)=", fopt(theta_old)    
    #print "f(theta_new)=", fopt(theta_new)    
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

### Classification for Logistic regression on majority voting & concatenating

def LogisticRegressionROC(Data,Label,TrueLabel, Color,legend, marker):
    Majority_Logistic = LogisticRegression()
    Majority_Logistic.verbose=0
    N_folds=5
    kf= cv.KFold(X.shape[0],n_folds=N_folds)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    for i, (train, test) in enumerate(kf):
        probas_ = Majority_Logistic.fit(Data[train], Label[train]).predict_proba(Data[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(TrueLabel[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        #roc_auc = auc(fpr, tpr)
        #plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
    
    mean_tpr /= len(kf)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    
    plt.plot(mean_fpr, mean_tpr, marker ,color=Color,label='%s AUC:%0.2f' % (legend,mean_auc), lw=2)
    
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
#X_concatenate=np.repeat(X,nbOfExperts,axis=0)
#Y_concatenate=np.reshape(Y,np.prod(Y.shape))

### ROC for EM-algorithm
fpr, tpr, thresholds = roc_curve(Z, expit(np.sum(np.multiply(X,np.tile(alpha,(X.shape[0],1))),1)+beta))
plt.plot(fpr, tpr, '-.',color=(1,0,1), label='Bernouilli AUC:%0.2f' % auc(fpr, tpr), lw=2)

plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))

LogisticRegressionROC(np.repeat(X,nbOfExperts,axis=0),np.reshape(Y,np.prod(Y.shape)),np.repeat(Z,nbOfExperts,axis=0),(0,0,1),"L.R. Concatenation",'-.')

LogisticRegressionROC(X,Y_maj,Z,(0,0,0),"L.R. Majority",'-')

for i in range(nbOfExperts):
    if i%2==0:
        Color=(1,1,0.31)
    else:
        Color=(0.39,1,0.33)
    LogisticRegressionROC(X,Y[:,i],Z,Color,"L.R.-Annotator{}".format(i+1),'--')

plt.show()

#################    Log Likelihood Ratio    ################# 
plt.bar(np.arange(1,nbOfExperts+1),[sum(clusters==t) for t in range(nbOfExperts)],align='center', color=[(0,1,0),(1,0,0),(1,1,0),(0,1,1),(1,0,1)])
plt.show()

stats1=[[abs(np.inner(kmeans.cluster_centers_[T,:],weight[:,t])+gamma[t]) for t in range(nbOfExperts)] for T in range(nbOfExperts)]
stats2=[[np.mean([abs(np.inner(X[i,:],weight[:,t])+gamma[t]) for i in np.where(clusters==T)]) for t in range(nbOfExperts)] for T in range(nbOfExperts)]
stats3=[[np.mean([ (-1)**(1-Y[i,t])*(np.inner(X[i,:],weight[:,t])+gamma[t]) for i in np.where(clusters==T)]) for t in range(nbOfExperts)] for T in range(nbOfExperts)]
stats4=[[np.mean([(-1)**abs(Z[i]-Y[i,t])*(np.inner(X[i,:],weight[:,t])+gamma[t]) for i in np.where(clusters==T)]) for t in range(nbOfExperts)] for T in range(nbOfExperts)]

def plotAnnotators(stats):
    for t in range(nbOfExperts):
        plt.bar(np.arange(1,nbOfExperts+1),stats[t],align='center')
        plt.show()

#plotAnnotators(stats4)
#################    Ground-Truth Estimation without features    ################# 

#np.sum([expit(np.inner(alpha,x)+beta)*np.prod([(1-expit(np.inner(weight[:,t],x)+gamma[t]))**(1-Y[i,t])+expit(np.inner(weight[:,t],x)+gamma[t])**Y[i,t] for t in range(nbOfExperts)]) for i,x in enumerate(X)])/X.shape[0]