# -*- coding: utf-8 -*-

"""
Created 01/06/2015

@author: PesneauReizine
"""

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
    z = w*(X.transpose())
    p = 1/(1+exp(-z))
    return p

def a(alpha,Y):
    result = np.array([Y.shape[0],1])
    for i in range(Y.shape[0]):
        y = Y[i,:]
        t_1 = alpha**y
        t_2 = (1-alpha)**(1-y)
        prod = t_1*t_2
        res = reduce(operator.mul,prod)
        result[i,0]=res
    return result

def b(beta,Y):
    return a(1-beta,Y)

###EM Algorithm
mu = (1/nbOfExperts)*sum(Y.transpose())
w = (1./5)*np.ones(5)
#alphanew = np.inner(mu.transpose(),Y)/sum(mu)
#betanew = (1-mu)*(1-Y)/sum(1-mu)



###Finir l'algorithme EM
##for k in range(1000):
##    ###E-step
##    a = a(alphanew,Y)
##    b = b(betanew,Y)
##    p = p(w,x)
    
 
    

















    
    
