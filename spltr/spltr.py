#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
``Spltr`` is a simple PyTorch-based data loader and splitter. 
It may be used to load arrays and matrices or Pandas DataFrames 
and CSV files containing numerical data with the subsequent split 
it into train, test (validation) subsets in the form of 
PyTorch DataLoader objects.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import torch
from torch.utils.data import TensorDataset, random_split, DataLoader
import pandas as pd
import numpy as np
from spltr.mods.process_variable import processVariable

class Spltr:
    """Loads numerical data and splits it into train, test (validation) 
    subsets in the form of PyTorch DataLoader objects.
         
    Arguments:
    ----------
    x|y : source of Input|Target data. May be i) a string with a link to 
        CSV file, ii) a Pandas DataFrame or iii) a list, array, matrice. 
        (!) x|y shall have the same length.
        
    Methods
    ----------
    `Spltr.process_x|y` : converts loaded Input/Target data into PyTorch 
        Tensors with ability to i) preview, ii) define tensor dtype, 
        iii) set the desired device of returned tensor (CPU/GPU), 
        iv) use selected columns from Input/Target data sources 
        or process a single data table (for CSV and Pandas DataFrame only).
           
    `Spltr.reshape_xy` : reshapes subsets.
    
    `Spltr.split_data` : splits data subsets into train, test (validation) 
        PyTorch DataLoader objects.
    
    `Spltr.clean_space` : optimizes memory by deleting unnecessary variables.
    """
    
    def __init__(self, x, y):       
        
        self.x = x
        self.y = y        
        
    def process_x(self, 
                  preview=False, 
                  tensor_dtype=None, 
                  tensor_device=None,                                  
                  usecols=None, 
                  **kwargs):
        '''
        Converts Input data into PyTorch Tensors.
        
        Arguments:
        ----------
        preview : set 'True' to preview 2 lines of the CSV/Pandas DataFrame being loaded.
        
        tensor_dtype : defines PyTorch Tensor dtype.
            Details: https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.dtype
        
        tensor_device : str, default 'CPU'.
            Set the desired device of returned tensor: 'CPU' or 'GPU'.
        
        csv_sep : [for CSV only] str, default ','
            Delimiter to use while loading CSV file.            
               
        usecols : [for CSV and Pandas DataFrame only] int, str, list or dict, optional.
            Return a subset of the columns to be converted into tensors.            
            i) selecting a single column by index: usecols = int
            ii) selecting a single column by name: usecols = str
            iii) selecting a number of columns by index: usecols = list(int, int, int)
            iv) selecting a number of columns by name: usecols = list(str, str, str)
            v) selecting a range of columns by index: usecols = dict(int[start]:int[end])
               -- worksonly for pd.DataFrames
            vi) selecting a range of columns by name: usecols = dict(str[start]:str[end])
               -- worksonly for pd.DataFrames
            
        kwargs : native commands from pd.csv_reader() are supported as well:            
            sep=',',
            delimiter=None,
            header='infer',
            names=None,
            index_col=None,
            usecols=None,
            squeeze=False,
            prefix=None,
            mangle_dupe_cols=True,
            dtype=None,
            engine=None,
            converters=None,
            true_values=None,
            false_values=None,
            skipinitialspace=False,
            skiprows=None,
            skipfooter=0,
            nrows=None,
            na_values=None,
            keep_default_na=True,
            na_filter=True,
            verbose=False,
            skip_blank_lines=True,
            parse_dates=False,
            infer_datetime_format=False,
            keep_date_col=False,
            date_parser=None,
            dayfirst=False,
            cache_dates=True,
            iterator=False,
            chunksize=None,
            compression='infer',
            thousands=None,
            decimal=b'.',
            lineterminator=None,
            quotechar='"',
            quoting=0,
            doublequote=True,
            escapechar=None,
            comment=None,
            encoding=None,
            dialect=None,
            error_bad_lines=True,
            warn_bad_lines=True,
            delim_whitespace=False,
            low_memory=True,
            memory_map=False,
            float_precision=None
        '''  
        
        self.x = processVariable(self.x, 
                                 preview=preview, 
                                 tensor_dtype=tensor_dtype, 
                                 tensor_device=tensor_device,                                                                 
                                 usecols=usecols,
                                 x_var=True, 
                                 **kwargs)        
        
            
    def process_y(self, 
                  preview=False, 
                  tensor_dtype=None, 
                  tensor_device=None,                                                     
                  usecols=None, 
                  **kwargs):
        '''
        Converts Target data into a PyTorch Tensor.
        
        Arguments:
        ----------
        preview : set 'True' to preview 2 lines of the CSV/Pandas DataFrame being loaded.
        
        tensor_dtype : defines PyTorch Tensor dtype.
            Details: https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.dtype
        
        tensor_device : str, default 'CPU'.
            Set the desired device of returned tensor: 'CPU' or 'GPU'.
        
        csv_sep : [for CSV only] str, default ','
            Delimiter to use while loading CSV file.        
        
        usecols : [for CSV and Pandas DataFrame only] int, str, list or dict, optional.
            Return a subset of the columns to be converted into tensors.            
            i) selecting a single column by index: esecols = int
            ii) selecting a single column by name: esecols = str
            iii) selecting a number of columns by index: esecols = list(int, int, int)
            iv) selecting a number of columns by name: esecols = list(str, str, str)
            v) selecting a range of columns by index: esecols = dict(int[start]:int[end])
               -- worksonly for pd.DataFrames
            vi) selecting a range of columns by name: esecols = dict(str[start]:str[end])
               -- worksonly for pd.DataFrames
            
        kwargs : native pd.csv_reader() arguments are supported as well:            
            sep=',',
            delimiter=None,
            header='infer',
            names=None,
            index_col=None,
            usecols=None,
            squeeze=False,
            prefix=None,
            mangle_dupe_cols=True,
            dtype=None,
            engine=None,
            converters=None,
            true_values=None,
            false_values=None,
            skipinitialspace=False,
            skiprows=None,
            skipfooter=0,
            nrows=None,
            na_values=None,
            keep_default_na=True,
            na_filter=True,
            verbose=False,
            skip_blank_lines=True,
            parse_dates=False,
            infer_datetime_format=False,
            keep_date_col=False,
            date_parser=None,
            dayfirst=False,
            cache_dates=True,
            iterator=False,
            chunksize=None,
            compression='infer',
            thousands=None,
            decimal=b'.',
            lineterminator=None,
            quotechar='"',
            quoting=0,
            doublequote=True,
            escapechar=None,
            comment=None,
            encoding=None,
            dialect=None,
            error_bad_lines=True,
            warn_bad_lines=True,
            delim_whitespace=False,
            low_memory=True,
            memory_map=False,
            float_precision=None
        '''     
        
        self.y = processVariable(self.y, 
                                 preview=preview, 
                                 tensor_dtype=tensor_dtype, 
                                 tensor_device=tensor_device,                                 
                                 usecols=usecols,
                                 x_var=False,                                  
                                 **kwargs)          
    
    
    def reshape_xy(self, x_shape=None, y_shape=None):
        """
        Reshapes Input|Target data subsets.
                
        Arguments:
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
    
    
    def split_data(self, splits=0.5, perm_seed = False, **kwargs):        
        """
        Splits data subsets into train, test (validation) PyTorch DataLoader objects.
        
        Arguments:
        ----------
        splits : int, list or tuple with 2 numerical values, default = 0.5
            Sets split ratio which shall be less than 1.0 (represents 100%)
            i) single int --> splits data into 2 dataloaders where Train = single int, 
                Test = 1.0 - single int
            ii) list|tuple(int #1, int #2) --> splits data into 3 dataloaders 
                where Train = int #1, Test = int #2 and Validation = 
                1.0 - int #1 - int #2. The sum of int #1 & int #2 
                shall be < 1.0       
                
        perm_seed : int, default = 'False', optional.
            Returns a random permutation of the datasets in the PyTorch DataLoader.        
         
        kwargs : native PyTorch DataLoader arguments are supported as well:
            batch_size=1,
            shuffle=False,
            sampler=None,
            batch_sampler=None,
            num_workers=0,
            collate_fn=None,
            pin_memory=False,
            drop_last=False,
            timeout=0,
            worker_init_fn=None,
            multiprocessing_context=None,        
        """    
        
        assert type(self.x) is torch.Tensor, "[X]: Expected PyTorch tensor as an input"
        assert type(self.y) is torch.Tensor, "[y]: Expected PyTorch tensor as an input"
        assert len(self.x) == len(self.y), "[X,y] should be of the same length"                
        
        # A random permutation of the datasets
        
        if perm_seed: 
            
            # Setting the seed for generating random order of dataset elements
            self.random_seed_ = perm_seed  
            assert type(self.random_seed_) is int, "Permutation Seed is expected to be an integer"            
            torch.manual_seed(self.random_seed_)
            
            # Permutation process
            self.x = self.x[torch.randperm(self.x.size()[0])]
            torch.manual_seed(self.random_seed_)
            self.y = self.y[torch.randperm(self.y.size()[0])]
        
        self.dataset_xy = TensorDataset(self.x, self.y)        

        self.db_length_ = len(self.x)  
        
        # Splitting a dataset by a single float

        if type(splits) is float:
            
            # Calculating splits            
            self.train_length_ = int(self.db_length_*splits)
            self.test_length_ = self.db_length_-self.train_length_
            self.val_length_ = 0
            self.splits_ = self.train_length_, self.test_length_

            self.xy_train, self.xy_test = random_split(self.dataset_xy, self.splits_)
            
            # Creating a dataloader objects
            self.xy_train = DataLoader(self.xy_train, **kwargs)
            self.xy_test = DataLoader(self.xy_test, **kwargs)             
            print(f'[X,y]: The Data is splited into 2 datasets of length: Train {self.train_length_}, Test {self.test_length_}.')

        # Splitting a dataset by a tuple or list    
            
        elif type(splits) is tuple or list:
            assert len(splits) < 3, "Too many split criterea are given. Expected no more than 2"

            # Safety checks as for the type of the input data            
            if len(splits) == 1:
                assert splits[0] <=1, "Split value shall not be greater 1"

                # Calculating splits
                self.train_length_ = int(self.db_length_*splits[0])
                self.test_length_ = self.db_length_-self.train_length_
                self.val_length_ = 0
                self.splits_ = self.train_length_, self.test_length_ 

                self.xy_train, self.xy_test = random_split(self.dataset_xy, self.splits_)                

                # Creating a dataloader objects                
                self.xy_train = DataLoader(self.xy_train, **kwargs)
                self.xy_test = DataLoader(self.xy_test, **kwargs) 
                print(f'[X,y]: The Data is splitted into 2 datasets of length: Train {self.train_length_}, Test {self.test_length_}.')

            else:
                # Calculating splits
                self.train_length_ = int(self.db_length_*splits[0])
                self.test_length_ = int(self.db_length_*splits[1])
                self.val_length_ = self.db_length_-self.train_length_-self.test_length_
                self.splits_ = self.train_length_, self.test_length_, self.val_length_          

                self.xy_train, self.xy_test, self.xy_val = random_split(self.dataset_xy, self.splits_)                

                # Creating a dataloader objects                
                self.xy_train = DataLoader(self.xy_train, **kwargs)
                self.xy_test = DataLoader(self.xy_test, **kwargs)
                self.xy_val = DataLoader(self.xy_val, **kwargs)                
                print(f'[X,y]: The Data is splitted into 3 datasets of length: Train {self.train_length_}, Test {self.test_length_}, Validation {self.val_length_}.')            
          
                
    def clean_space(self):
        """
        Optimize memory by deleting unnecessary variables.
        Advised to be performed once no class functions are to be called.
        """
        
        del(self.x, self.y, self.dataset_xy, self.db_length_, self.train_length_, self.test_length_, self.val_length_, self.splits_)
            
        print('All variables are deleted. Only Train-Test (Validation) sets are left.')

