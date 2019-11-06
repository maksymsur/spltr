#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import torch
import pandas as pd
import numpy as np

def processVariable(x, preview=False, 
                    tensor_dtype=None, 
                    tensor_device=None, 
                    x_var=None,
                    usecols=None,                    
                    **kwargs):
    
    
    # Identifying a type of the dataset we are dealing with: [X] or [y]
    
    if x_var == True:        
        idVar = "[X]"          
    else:                
        idVar = "[y]"
    
    # Reading a CSV file providing that the file location is indicated 
    
    if type(x) is str:
        x = pd.read_csv(x, usecols=usecols, **kwargs)        
        
        if preview:
            print(idVar + ": The following CSV was loaded:", x.head(2), sep="\n\n", end="\n\n")
        
        # Conversion a CSV into a PyTorch Tensor         
        x = torch.tensor(x.values, dtype=tensor_dtype, device=tensor_device)            
        
    # Reading a Pandas DataFrame    
    
    elif type(x) is (pd.core.frame.DataFrame or pd.core.series.Series):
        
        # Columns extraction        
        if type(usecols) is int:
            x = x.iloc[:,usecols]

        elif type(usecols) is str:
            x = x.loc[:,usecols]

        elif type(usecols) is list:
            
            if type(usecols[0]) is int:
                x = x.iloc[:,usecols]

            elif type(usecols[0]) is str:
                x = x.loc[:,usecols]

        elif type(usecols) is dict:
            
            _temp1 = [i for i in usecols.items()]
            
            if (type(_temp1[0][0]) is int) and (type(x) is pd.core.series.Series):
                x = x[_temp1[0][0]:_temp1[0][1]].T
                
            elif (type(_temp1[0][0]) is int) and (type(x) is pd.core.frame.DataFrame):
                x = x.iloc[:,_temp1[0][0]:_temp1[0][1]]
                
            elif type(_temp1[0][0]) is str:
                x = x.loc[:,_temp1[0][0]:_temp1[0][1]]

            del(_temp1)
        
        if preview:
            print(idVar + ": The following DataFrame was loaded:", x.head(2), sep="\n\n", end="\n\n")
            
        # Conversion a Pandas DataFrame into a PyTorch Tensor         
        x = torch.tensor(x.values, dtype=tensor_dtype, device=tensor_device)                          

        
    elif type(x) is list or tuple:        
        x = torch.tensor(x, dtype=tensor_dtype, device=tensor_device)

    elif type(x) is np.ndarray:        
        x = torch.tensor(x, dtype=tensor_dtype, device=tensor_device)
            
    elif type(x) is torch.Tensor:        
        x = x.to(tensor_dtype).to(tensor_device)

    else:        
        raise ValueError(idVar + ": Not supported data type for input")

    print(idVar + ": Tensor of shape {} was created".format(x.size()))
    
    return x

