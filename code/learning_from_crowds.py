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

###Every array is in column
##Y is a (350,5) matrix
##X is a (350,34) matrix
##Z is a (350,1) vector
##mu is a (350,1) vector

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

###Building data
X,Z = load_ionosphere("ionosphere.txt")
alpha = np.arange(0.75,1,0.05)#true positive rate
beta = np.arange(0.5,0.75,0.05)#1-false positive rate
##build noisy labels based on Two-coin model for annotators
Y = np.ones([Z.shape[0],nbOfExperts])
for i in range(Y.shape[0]):
    r = 100*random()
    z = Z[i]
    if(z):
        for j in range(Y.shape[1]):
            r = random()
            Y[i,j] = (r<alpha[j])
    else:
        for j in range(Y.shape[1]):
            r = random()
            Y[i,j] = (r>beta[j])


def p(w,x):#x a nbOfExperts colonnes et 34 lignes
    z = np.inner(X,w)
    p = np.ones(z.shape[0])
    for i in range(z.shape[0]):
        p[i] = 1/(1+exp(-z[i]))
    return p

def a(alpha,Y):
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

def b(beta,Y):
    return a(1-beta,Y)

    
def gradp(x,mu,w):
    return tools.to_col(np.inner(tools.to_line(mu-p(w,x)),x.transpose()))

def hessianp(x,w):
    result=0
    sigma=p(w,x)
    for i in range(x.shape[1]):
        xinte=x[i,:]
        sigmatemp=sigma[i]
        result+=sigmatemp*(1-sigmatemp)*np.dot(tools.to_col(xinte),tools.to_line(xinte))
    return result



###EM Algorithm
mu = (1./nbOfExperts)*sum(Y.transpose())
w = (1./34)*np.ones(34)
alphanew = np.inner(tools.to_line(mu),Y.transpose())/sum(mu)
betanew = np.inner(tools.to_line(1-mu),1-Y.transpose())/sum(1-mu)



def f(w,X,alpha,Y,beta,mu):
    aa=tools.to_line(a(alpha,Y))
    bb=tools.to_line(b(beta,Y))
    pp=tools.to_line(p(w,X))
    muu=tools.to_line(mu)
    ln = np.zeros((1,aa.shape[1]))
    prod = aa*pp
    for i in range(aa.shape[1]):
        ln[0,i] = log(prod[0,i])
    ln_2 = np.zeros((1,aa.shape[1]))
    for i in range(aa.shape[1]):
        ln_2[0,i] = log(1-pp[0,i])
    fopt = muu*ln+(1-mu)*ln_2*bb
    fopt = reduce(operator.mul,fopt)
    fopt = sum(fopt)
    return fopt


def fopt(w):
    return f(w,X,alphanew,Y,betanew,mu)

def gradopt(w):
    return gradp(X,mu,w)

def hessopt(w):
    return hessianp(X,w)

#optimize.newton(func=fopt,x0=w,fprime=gradopt)#,fprime2=hessopt)
print "taille de w : ",w.shape
optimize.minimize(fun=fopt,x0=w,method='Newton-CG',jac=gradopt,hess=hessopt)
###Finir l'algorithme EM
##for k in range(1000):
##    ###E-step
##    print "E-step"
##    a = a(alphanew,Y)
##    b = b(betanew,Y)
##    p = p(w,x)
##    ##Calculer mu
##
##    ###M-step
##    alphanew = np.inner(tools.to_line(mu),Y.transpose())/sum(mu)
##    betanew = np.inner(tools.to_line(1-mu),1-Y.transpose())/sum(1-mu)
##    ##Calculer w

    

















    
    
