# -*- coding: utf-8 -*-

"""
Created 01/06/2015

@author: PesneauReizine
"""

import tools
import numpy as np # module pour les outils mathématiques
import matplotlib.pyplot as plt # module pour les outils graphiques
from sklearn.cluster import KMeans #module pour trouver des clusters par la méthode k-means
from scipy import optimize #module pour optimiser des fonction
from scipy.special import expit
from math import exp,log
from functools import reduce
import operator
from random import random
from sklearn import cross_validation as cv
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from scipy import interp
###Every array is in column
##Y is a (350,5) matrix
##X is a (350,34) matrix
##Z is a (350,1) vector
##mu is a (350,1) vector

nbOfExperts = 5
iter_max=100
tolerance=0.0001
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

###Building data
X,Z = load_ionosphere("../DataSets/ionosphere.txt")
alpha = np.arange(0.4,0.9,0.1)#true positive rate
beta = np.arange(0.3,0.75,0.1)#1-false positive rate
##build noisy labels based on Two-coin model for annotators
Y = np.ones([Z.shape[0],nbOfExperts])
for i in range(Y.shape[0]):
    
    z = Z[i]
    if(z):
        for j in range(Y.shape[1]):
            r = random()
            Y[i,j] = (r<alpha[j])
    else:
        for j in range(Y.shape[1]):
            r = random()
            Y[i,j] = (r>beta[j])


def compute_p(w,x):#x a nbOfExperts colonnes et 34 lignes
    z = np.inner(X,w)
    p = np.ones(z.shape[0])
    for i in range(z.shape[0]):
        p[i] = 1/(1+exp(-z[i]))
    return p

def compute_a(alpha,Y):
    result = np.ones(Y.shape[0])
    for i in range(Y.shape[0]):
        y = Y[i,:]
        t_1 = alpha**y
        t_2 = (1-alpha)**(1-y)
        prod = t_1*t_2
        res = reduce(operator.mul,prod)
        res = reduce(operator.mul,res)
        result[i]=res
    return result

def compute_b(beta,Y):
    return compute_a(1-beta,Y)

def f(w,X,alpha,Y,beta,mu):
    aa=tools.to_line(compute_a(alpha,Y))
    bb=tools.to_line(compute_b(beta,Y))
    pp=tools.to_line(compute_p(w,X))
    muu=tools.to_line(mu)
    ln = np.zeros((1,aa.shape[1]))
    prod = aa*pp
    for i in range(aa.shape[1]):
        ln[0,i] = log(prod[0,i])
    ln_2 = np.zeros((1,aa.shape[1]))
    for i in range(aa.shape[1]):
        try:
            ln_2[0,i] = log(1-pp[0,i])
        except ValueError:
            #print("là il se passe un truc")
            pass
    fopt = muu*ln+(1-mu)*ln_2*bb
    fopt = reduce(operator.mul,fopt)
    fopt = sum(fopt)
    return fopt

def gradp(x,mu,w):
    return tools.to_col(np.inner(tools.to_line(mu-compute_p(w,x)),x.transpose()))

def hessianp(x,w):
    result=0
    sigma=compute_p(w,x)
    for i in range(x.shape[1]):
        xinte=x[i,:]
        sigmatemp=sigma[i]
        result+=sigmatemp*(1-sigmatemp)*np.dot(tools.to_col(xinte),tools.to_line(xinte))
    return result



###EM Algorithm
iter=0
ecart=10


while iter<iter_max and ecart>tolerance:
###  E Step
    iter=iter+1
    if iter==1:
        mu = (1./nbOfExperts)*sum(Y.transpose())
        w = (1./X.shape[1])*np.ones(X.shape[1])
        
    else:
        a=compute_a(alphanew,Y)
        b=compute_b(betanew,Y)
        p=compute_p(w,X)
        ap=a*p
        mu=ap/(ap+b*(1-p))






### M Step
    if iter>1:
        alphaold=alphanew
        betaold=betanew
        alphanew = np.inner(tools.to_line(mu),Y.transpose())/sum(mu)
        betanew = np.inner(tools.to_line(1-mu),1-Y.transpose())/sum(1-mu)
        print("ecart",ecart)
        ecart=max(np.linalg.norm(alphaold-alphanew),np.linalg.norm(betaold-betanew))
    else:
        alphanew = np.inner(tools.to_line(mu),Y.transpose())/sum(mu)
        betanew = np.inner(tools.to_line(1-mu),1-Y.transpose())/sum(1-mu)
    def negf(w):
    
        return -f(w,X,alphanew,Y,betanew,mu)
    

    def neggrad(w):
    #si on enleve le [0,:], on reçoit une erreur d'incompatiblité des dimensions (34,34) avec (1,34)
        return -gradp(X,mu,w)[0,:]

    def neghess(w):
        return -hessianp(X,w)

    bnds=w.shape[0]*[(0,10**8)]


    res = optimize.minimize(negf, w,bounds=bnds,jac=neggrad,hess=neghess,method='Newton-CG',options={'disp': False, 'gtol':1})
    w=res.x
    print("Iteration : ",iter)
print("alpha real",alpha)
print("alpha found ",alphanew)
print("beta real",beta)
print("beta found ", betanew)



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
fpr, tpr, thresholds = roc_curve(Z, mu)
plt.plot(fpr, tpr, '-.',color=(1,0,1), label='Bayesian approach:%0.2f' % auc(fpr, tpr), lw=2)

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
    

















    
    
