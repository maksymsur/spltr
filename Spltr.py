#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
``Spltr`` is a simple PyTorch-based data loader and splitter. 
It may be used to load arrays and matrices or Pandas DataFrames 
and CSV files containing numerical data with the subsequent split it into 
train, test (validation) subsets in the form of PyTorch DataLoader objects.
"""

import torch
from torch.utils import data
import numpy as np
import pandas as pd

class Spltr:
    """Load arrays and matrices or Pandas DataFrames and CSV files 
    containing numerical data and split it into train, test (validation) 
    subsets in the form of PyTorch DataLoader objects.   
         
    Parameters
    ----------
    x : source of Training data. May be i) a string with the link to CSV file, 
        ii) a Pandas DataFrame or iii) list, array, matrice with the same 
        length/shape as 'y'.
    y : source of Target data. May be i) a string with the link to CSV file, 
        ii) a Pandas DataFrame or iii) list, array, matrice with the same 
        length/shape as 'x'. 
        
    Methods
    ----------
    Spltr.process_x|y : converts loaded Training/Target data into PyTorch Tensors 
        with ability to i) preview, ii) define tensor dtype, iii) set the desired 
        device of returned tensor (CPU/GPU), iv) use selected rows/columns from 
        Training/Target data sources or a single data table for processing 
        (for CSV and Pandas DataFrame only).
           
    Spltr.reshape_xy : reshapes subsets.
    
    Spltr.split_data : splits data subsets into train, test (validation) 
        PyTorch DataLoader objects.
    
    Spltr.clean_space : optimizes mamory by deleting unnecessary variables.
    """
    
    def __init__(self, x, y):       
                        
        self.x = x
        self.y = y
        
    def process_x(self, preview=False, tensor_dtype=None, tensor_device=None, csv_sep=',', header=0, nrows=None, usecols=None):
        '''
        Converts Training data into a PyTorch Tensor.
        
        Parameters
        ----------
        preview : ability to preview 2 lines of CSV/Pandas DataFrame being loaded.
        
        tensor_dtype : defines PyTorch Tensor dtype.
            Details: https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.dtype
        
        tensor_device : str, default 'CPU'.
            Set the desired device of returned tensor: 'CPU' or 'GPU'.
        
        csv_sep : [for CSV only] str, default ','
            Delimiter to use while loading CSV file.
            
        header=None
        
        nrows : [for CSV and Pandas DataFrame only] int, optional.
            Number of rows to read. Useful for reading pieces of large files.
        
        usecols : [for CSV and Pandas DataFrame only] int, str, list or dict, optional.
            Return a subset of the columns to be converted into tensors.            
            i) selecting a single column by index: usecols = int
            ii) selecting a single column by name: usecols = str
            iii) selecting a number of columns by index: usecols = list(int, int, int)
            iv) selecting a number of columns by name: usecols = list(str, str, str)
            v) selecting a range of columns by index: usecols = dict(int[start]:int[end])
            vi) selecting a range of columns by name: usecols = dict(str[start]:str[end])
        '''  
        
        if type(self.x) is str:
            assert self.x[0] == '/', "[X]: String should contain link to file, not text"
            self.x = pd.read_csv(self.x, sep=csv_sep, header=header, nrows=nrows, usecols=usecols)
            if preview:
                print('[X]: The following CSV was loaded:', self.x.head(2), sep='\n\n', end='\n\n')
            self.x = torch.tensor(self.x.values, dtype=tensor_dtype, device=tensor_device)            

        elif type(self.x) is (pd.core.frame.DataFrame or pd.core.series.Series):            
                                  
            # Columns extraction
            
            if type(usecols) is int:
                self.x = self.x.iloc[:,usecols]

            elif type(usecols) is str:
                self.x = self.x.loc[:,usecols]

            elif type(usecols) is list:

                if type(usecols[0]) is int:
                    self.x = self.x.iloc[:,usecols]

                elif type(usecols[0]) is str:
                    self.x = self.x.loc[:,usecols]

            elif type(usecols) is dict:

                self._temp1 = [i for i in usecols.items()]

                if (type(self._temp1[0][0]) is int) and (type(self.x) is pd.core.series.Series):
                    self.x = self.x[self._temp1[0][0]:self._temp1[0][1]].T
                
                elif (type(self._temp1[0][0]) is int) and (type(self.x) is pd.core.frame.DataFrame):
                    self.x = self.x.iloc[:,self._temp1[0][0]:self._temp1[0][1]]
                
                elif type(self._temp1[0][0]) is str:
                    self.x = self.x.loc[:,self._temp1[0][0]:self._temp1[0][1]]

                del(self._temp1)
                        
            # Rows extraction
            
            if type(nrows) is int:
                self.x = self.x.iloc[nrows,:]           

            elif type(nrows) is list:
                assert type(nrows[0]) is int, 'List shall contain integers separated by comas to define rows for selection.'
                self.x = self.x.iloc[nrows,:]                

            elif type(nrows) is dict:
                self._temp2 = [i for i in nrows.items()]

                if (type(self._temp2[0][0]) is int) and (type(self.x) is pd.core.series.Series):
                    self.x = self.x[self._temp2[0][0]:self._temp2[0][1]].T
                
                elif (type(self._temp2[0][0]) is int) and (type(self.x) is pd.core.frame.DataFrame):
                    self.x = self.x.iloc[self._temp2[0][0]:self._temp2[0][1],:]               

                del(self._temp2)
            
            if preview:
                print('[X]: The following DataFrame was loaded:', self.x.head(2), sep='\n\n', end='\n\n')
            
            # Conversion into a PyTorch Tensor            
            self.x = torch.tensor(self.x.values, dtype=tensor_dtype, device=tensor_device)                          

        elif type(self.x) is list or tuple:
            self.x = torch.tensor(self.x, dtype=tensor_dtype, device=tensor_device)

        elif type(self.x) is np.ndarray:
            self.x = torch.tensor(self.x, dtype=tensor_dtype, device=tensor_device)
            
        elif type(self.x) is torch.Tensor:
            self.x = self.x.to(tensor_dtype).to(tensor_device)

        else:
            raise ValueError('[X]: Not supported data type for input')

        print(f'[X]: Tensor of shape {self.x.size()} was created')
        
            
    def process_y(self, preview=False, tensor_dtype=None, tensor_device=None, csv_sep=',', header=0, nrows=None, usecols=None):
        '''
        Converts Target data into a PyTorch Tensor.
        
        Parameters
        ----------
        preview : ability to preview 2 lines of CSV/Pandas DataFrame being loaded.
        
        tensor_dtype : defines PyTorch Tensor dtype.
            Details: https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.dtype
        
        tensor_device : str, default 'CPU'.
            Set the desired device of returned tensor: 'CPU' or 'GPU'.
        
        csv_sep : [for CSV only] str, default ','
            Delimiter to use while loading CSV file.
        
        nrows : [for CSV and Pandas DataFrame only] int, optional.
            Number of rows to read. Useful for reading pieces of large files.
        
        usecols : [for CSV and Pandas DataFrame only] int, str, list or dict, optional.
            Return a subset of the columns to be converted into tensors.            
            i) selecting a single column by index: esecols = int
            ii) selecting a single column by name: esecols = str
            iii) selecting a number of columns by index: esecols = list(int, int, int)
            iv) selecting a number of columns by name: esecols = list(str, str, str)
            v) selecting a range of columns by index: esecols = dict(int[start]:int[end])
            vi) selecting a range of columns by name: esecols = dict(str[start]:str[end])
        '''     
        
        if type(self.y) is str:
            assert self.y[0] == '/', "[y]: String should contain link to file, not text"
            self.y = pd.read_csv(self.y, sep=csv_sep, header=header, nrows=nrows, usecols=usecols)
            if preview:
                print('[y]: The following CSV was loaded:', self.y.head(2), sep='\n\n', end='\n\n')
            self.y = torch.tensor(self.y.values, dtype=tensor_dtype, device=tensor_device)            

        elif type(self.y) is (pd.core.frame.DataFrame or pd.core.series.Series):            
                        
            # Columns extraction
            
            if type(usecols) is int:
                self.y = self.y.iloc[:,usecols]

            elif type(usecols) is str:
                self.y = self.y.loc[:,usecols]

            elif type(usecols) is list:

                if type(usecols[0]) is int:
                    self.y = self.y.iloc[:,usecols]

                elif type(usecols[0]) is str:
                    self.y = self.y.loc[:,usecols]

            elif type(usecols) is dict:

                self._temp3 = [i for i in usecols.items()]

                if (type(self._temp3[0][0]) is int) and (type(self.y) is pd.core.series.Series):
                    self.y = self.y[self._temp3[0][0]:self._temp3[0][1]].T
                
                elif type(self._temp3[0][0]) is int and (type(self.y) is pd.core.frame.DataFrame):
                    self.y = self.y.iloc[:,self._temp3[0][0]:self._temp3[0][1]]

                elif type(self._temp3[0][0]) is str:
                    self.y = self.y.loc[:,self._temp3[0][0]:self._temp3[0][1]]

                del(self._temp3)
                        
            # Rows extraction 
            
            if type(nrows) is int:
                self.y = self.y.iloc[nrows,:]           

            elif type(nrows) is list:
                assert type(nrows[0]) is int, 'List shall contain integers separated by comas to define rows for selection.'
                self.y = self.y.iloc[nrows,:]                

            elif type(nrows) is dict:                
                self._temp4 = [i for i in nrows.items()]

                if (type(self._temp4[0][0]) is int) and (type(self.y) is pd.core.series.Series):
                    self.y = self.y[self._temp4[0][0]:self._temp4[0][1]].T
                
                elif type(self._temp4[0][0]) is int and (type(self.y) is pd.core.frame.DataFrame):
                    self.y = self.y.iloc[self._temp4[0][0]:self._temp4[0][1],:]               

                del(self._temp4)
            
            if preview:
                print('[y]: The following DataFrame was loaded:', self.y.head(2), sep='\n\n', end='\n\n')
            
            # Conversion into a PyTorch Tensor            
            self.y = torch.tensor(self.y.values, dtype=tensor_dtype, device=tensor_device)                          

        elif type(self.y) is list or tuple:
            self.y = torch.tensor(self.y, dtype=tensor_dtype, device=tensor_device)

        elif type(self.y) is np.ndarray:
            self.y = torch.tensor(self.y, dtype=tensor_dtype, device=tensor_device)
            
        elif type(self.y) is torch.Tensor:
            self.y = self.y.to(tensor_dtype).to(tensor_device)

        else:
            raise ValueError('[y]: Not supported data type for input')

        print(f'[y]: Tensor of shape {self.y.size()} was created')
        
        # Sanity check to see if we have Training and Target subsets of the same length in the end.        
        assert len(self.x) == len(self.y), "Attention: [X] and [y] are of different length."

    def reshape_xy(self, x_shape=None, y_shape=None):
        """
        Reshapes Train|Target data subsets.
                
        Parameters
        ----------
        x|y_shape : int, list or tuple, optional. Similar to PyTorch Tensor.view()
            Returns a new tensor with the same data as the self tensor but of a different shape.
            Details: https://pytorch.org/docs/stable/tensors.html?highlight=view#torch.Tensor.view        
        """     
        
        if x_shape:
            assert type(x_shape) is int or list or tuple, "[x_shape]: An integer, list or tuple should be an input"            
            self.x = self.x.view(x_shape)
            self.shape_x_ = self.x.size()
        
        if y_shape:
            assert type(y_shape) is int or list or tuple, "[y_shape]: An integer, list or tuple should be an input"            
            self.y = self.y.view(y_shape)
            self.shape_y_ = self.y.size()
        
#         # Sanity check to see if we have Training and Target subsets of the same length in the end.        
#         assert len(self.x) == len(self.y), "Attention: [X] and [y] are of different length."
        
    
    def split_data(self, splits=0.5, batch_size=1, perm_seed = False, loader_shuffle=False):        
        """
        Splits data subsets into train, test (validation) PyTorch DataLoader objects.
        
        Parameters
        ----------
        splits : int, list or tuple with 2 numerical values, default = 0.5
            Sets split ratio which shall be under 1.0 (represents 100%)
            i) int --> splits data into 2 dataloaders where Train = int, 
                Test = 1.0 - int
            ii) list|tuple(int #1, int #2) --> splits data into 3 dataloaders 
                where Train = int #1, Test = int #2 and Validation = 
                1.0 - int #1 - int #2. The sum of int #1 & int #2 
                shall be < 1.0
        
        batch_size : int, default = 1.
            Set size of batches.
        
        perm_seed : int, default = 'False', optional.
            Returns a random permutation of the datasets in the PyTorch DataLoader.
        
        loader_shuffle: bool, default = 'False', optional.
            Set to `True` to have the data reshuffled at every epoch.
        
        """    
        
        assert type(self.x) is torch.Tensor, "[X]: Expected PyTorch tensor as an input"
        assert type(self.y) is torch.Tensor, "[y]: Expected PyTorch tensor as an input"
        assert len(self.x) == len(self.y), "[X,y] should be of the same length"
        
        if perm_seed:
            self.random_seed_ = perm_seed
            assert type(self.random_seed_) is int, "Permutation Seed is expected to be an integer"            
            torch.manual_seed(self.random_seed_)
            self.x = self.x[torch.randperm(self.x.size()[0])]
            torch.manual_seed(self.random_seed_)
            self.y = self.y[torch.randperm(self.y.size()[0])]
        
        self.dataset_xy = data.TensorDataset(self.x, self.y)        

        self.db_length_ = len(self.x)    

        if type(splits) is float:
            self.train_length_ = int(self.db_length_*splits)
            self.test_length_ = self.db_length_-self.train_length_
            self.val_length_ = 0
            self.splits_ = self.train_length_, self.test_length_

            self.xy_train, self.xy_test = data.random_split(self.dataset_xy, self.splits_)

            self.xy_train = data.DataLoader(self.xy_train, batch_size=batch_size, shuffle=loader_shuffle)
            self.xy_test = data.DataLoader(self.xy_test, batch_size=batch_size, shuffle=loader_shuffle)             
            print(f'[X,y]: The Data is splited into 2 datasets of length: Train {self.train_length_}, Test {self.test_length_}.', f'       DataLoaders with Batch Size {batch_size} and Shufle {loader_shuffle} are created', sep='\n')

        elif type(splits) is tuple or list:
            assert len(splits) < 3, "Too many split criterea are given. Expected no more than 2"

            if len(splits) == 1:
                assert splits[0] <=1, "Split value shall not be greater 1"

                self.train_length_ = int(self.db_length_*splits[0])
                self.test_length_ = self.db_length_-self.train_length_
                self.val_length_ = 0
                self.splits_ = self.train_length_, self.test_length_ 

                self.xy_train, self.xy_test = data.random_split(self.dataset_xy, self.splits_)                

                self.xy_train = data.DataLoader(self.xy_train, batch_size=batch_size, shuffle=loader_shuffle)
                self.xy_test = data.DataLoader(self.xy_test, batch_size=batch_size, shuffle=loader_shuffle) 
                print(f'[X,y]: The Data is splited into 2 datasets of length: Train {self.train_length_}, Test {self.test_length_}.', f'       DataLoaders with Batch Size {batch_size} and Shufle {loader_shuffle} are created', sep='\n')

            else:
                self.train_length_ = int(self.db_length_*splits[0])
                self.test_length_ = int(self.db_length_*splits[1])
                self.val_length_ = self.db_length_-self.train_length_-self.test_length_
                self.splits_ = self.train_length_, self.test_length_, self.val_length_          

                self.xy_train, self.xy_test, self.xy_val = data.random_split(self.dataset_xy, self.splits_)                

                self.xy_train = data.DataLoader(self.xy_train, batch_size=batch_size, shuffle=loader_shuffle)
                self.xy_test = data.DataLoader(self.xy_test, batch_size=batch_size, shuffle=loader_shuffle)
                self.xy_val = data.DataLoader(self.xy_val, batch_size=batch_size, shuffle=loader_shuffle)                
                print(f'[X,y]: The Data is splited into 3 datasets of length: Train {self.train_length_}, Test {self.test_length_}, Validation {self.val_length_}.', f'       DataLoaders with Batch Size {batch_size} and Shufle {loader_shuffle} are created', sep='\n')            
          
                
    def clean_space(self):
        """
        Optimize mamory by deleting unnecessary variables.
        Advised to be performed once no class functions are to be called.
        """
        
        del(self.x, self.y, self.dataset_xy, self.db_length_, self.train_length_, self.test_length_, self.val_length_, self.splits_)
            
        print('All variables are deleted. Only Train-Test(Validation) data is left')

