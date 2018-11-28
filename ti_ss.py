# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 16:31:24 2018
@author: John Siryj
"""
import numpy as np
import pandas as pd


#######################################
### log(x+y)=logplus(log(x),log(y)) ###
#######################################

def logplus(x, y):
    if x > y:
        return x + np.log(1 + np.exp(y - x))
    else:
        return y + np.log(1 + np.exp(x - y))

# logplus function for a vector x
def logplusvec(x):
    r = -np.inf
    for i in x:
        r = logplus(r, i)
    return r

"""
###################################
### Evidence estimation methods ###
###################################

 For both functions
  x is a Pandas DataFrame of the log-likelihood and the temperatures used
  for ss function, the temperatures must be sorted.
  temp: specification of temperatures to be used, for instance, [0,2,9],
        "None" stands for all of them
"""

# thermodynamic integration

def ti(x, actPlot = False, temp = None):
    if temp is not None: # selecting certain temperatures (temps)
        count = x.groupby('invTemp').aggregate([len]) # not sure why this is included in the function, left in for total translation
        index = pd.DataFrame(np.where(np.diff(x['invTemp']) != 0)[0])
        index = pd.concat([pd.concat([pd.DataFrame([0]),index+1]).reset_index(drop=True),pd.concat([index,pd.DataFrame([x.shape[0]-1])]).reset_index(drop=True)],axis=1) 
        index.columns = [0,1]
        index = index.iloc[temp]
        index = list(np.hstack(index.apply(lambda x: list(range(x[0],x[1]+1,1)),axis=1)))
        newX = x.iloc[index].reset_index(drop=True)
        x = newX
        
    Rti = x.groupby('invTemp').aggregate([np.mean]) # Mean per temperature
    Rti = Rti['logL']['mean'].reset_index()
    Rti.columns = ['invTemp','logL']
    
    return sum(np.diff(Rti['invTemp'])*(Rti['logL'][:-1] + Rti['logL'][1:].reset_index(drop=True))/2)

# steppingstone sampling        
    
def ss(x, temp = None):
    if temp is not None: # selecting certain temperatures (temps)
        count = x.groupby('invTemp').aggregate([len])
        index = pd.DataFrame(np.where(np.diff(x['invTemp']) != 0)[0])
        index = pd.concat([pd.concat([pd.DataFrame([0]),index+1]).reset_index(drop=True),pd.concat([index,pd.DataFrame([x.shape[0]-1])]).reset_index(drop=True)],axis=1) 
        index.columns = [0,1]
        index = index.iloc[temp]
        index = list(np.hstack(index.apply(lambda x: list(range(x[0],x[1]+1,1)),axis=1)))
        newX = x.iloc[index].reset_index(drop=True)
        x = newX
        
    count = x.groupby('invTemp').aggregate([len])
    # 'count' could be sorted here, increasing order, if 'x' is not ordered 
    count = count['logL']['len'].reset_index()
    count.columns = ['invTemp','length']
    diff = np.diff(count['invTemp']) #difference between temperatures
    count = count['length'][:-1].astype(int) # Number of elements per temperature/no posterior
    delta = np.repeat(diff,count) # replicating diff in original data
    N = sum(count) # = len(delta)
    logl = x['logL'][:N] # extracting loglike
    logldelta = logl*delta # loglike x delta
    temperature = x['invTemp'][:N] # extracting temperature
    Rss = pd.concat([temperature,logldelta],axis=1)
    Rss = Rss.groupby('invTemp').aggregate([logplusvec])
    Rss = Rss['logL']['logplusvec'].reset_index(drop=True) - np.log(count) # Individual rate estimates
    
    return sum(Rss)
      
