import torch
import numpy as np
from pympler import asizeof
# from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
from utils.pca_gen import *
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer

from sklearn.decomposition import FactorAnalysis

import torch.nn.functional as F


def getPCA(flatted_models):
    #centerlize
    # flatted_models=StandardScaler().fit_transform(flatted_models)
    # nan_clear=SimpleImputer(strategy='mean')
    # flatted_models=nan_clear.fit_transform(flatted_models)
    # print(flatted_models)
    
    # init PCA
    pca=PCA(n_components=2)
    
    # init PPCA
    # ppca=FactorAnalysis(n_components=2)
    
    principalCompents=pca.fit_transform(flatted_models)

    return principalCompents

def drawPAC(principalComopents,posi_list):
    color = ['red', 'black','blue']
    # principalDf=pd.DataFrame(data=principalComopents,columns=['c1','c2'])

    #get coordinate-wise medians
    # principalComopents=F.normalize(torch.FloatTensor(principalComopents))
    median=np.median(principalComopents,axis=0)


    for i in range(len(principalComopents)):
        if i in posi_list:
            plt.scatter(principalComopents[i][0], principalComopents[i][1], color=color[1])
        else:
            plt.scatter(principalComopents[i][0], principalComopents[i][1], color=color[0])
    plt.scatter(median[0],median[1],color=color[2])
    plt.show()