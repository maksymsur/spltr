# What is it?

**`Spltr`** is a simple PyTorch-based data loader and splitter.
It may be used to load i) arrays and ii) matrices or iii) Pandas 
DataFrames and iv) CSV files containing numerical data with
subsequent split it into `Train, Test (Validation)` subsets in
the form of `PyTorch DataLoader` objects. The special emphesis was 
given to ease of usage and automation of some frequent data-handling procedures.

Originally it was developed in order to speed up a data preparation stage
for number of trivial ML tasks. Hope it may be useful for you as well.

# Main Features

`Spltr.process_x|y` : converts loaded Input/Target data into PyTorch Tensors 
    with ability to i) preview, ii) define tensor dtype, iii) set the desired 
    device of returned tensor (CPU/GPU), iv) use selected rows/columns from 
    Input/Target data sources or process a single data table (for CSV and 
    Pandas DataFrame only).
           
`Spltr.reshape_xy` : reshapes subsets.
    
`Spltr.split_data` : splits data subsets into train, test (validation) 
    PyTorch DataLoader objects.
    
`Spltr.clean_space` : optimizes memory by deleting unnecessary variables.

# Installation

```python
pip install spltr
```

# License

OSI Approved :: MIT License

# Documentation

https://github.com/maksymsur/Spltr

# Dependencies

+ torch >= 1.1.0
+ numpy >= 1.16.4
+ pandas >= 0.24.2

# What's new

* **Version 0.3.2** brings i) bug fixes and ii) extension of `.split_data` method by including native DataLoader methods like `num_workers`, `pin_memory`, `worker_init_fn`, `multiprocessing_context` and others as per PyTorch documentation.

# Example of usage

Hereunder we'll build a simple neural network and describe how `Spltr` may be used in the process.

**STEP 1:** Let's start with loading and reading an Iris dataset to be used as an example. The dataset may be found at: https://github.com/maksymsur/Spltr/blob/master/dataset/iris_num.csv


```python
import pandas as pd

link = 'https://raw.githubusercontent.com/maksymsur/spltr/master/dataset/iris_num.csv'
db = pd.read_csv(link)
print(db.info())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 150 entries, 0 to 149
    Data columns (total 5 columns):
    sepal.length    150 non-null float64
    sepal.width     150 non-null float64
    petal.length    150 non-null float64
    petal.width     150 non-null float64
    variety         150 non-null int64
    dtypes: float64(4), int64(1)
    memory usage: 6.0 KB
    None


By building a neural network we'll try to predict a 'variety' column. This column identifies Iris species: Setosa = 0, Versicolor = 1 and Virginica = 2. Let's verify that it is comprised from the mentioned 3 unique values.


```python
print(db.variety.unique())
```

    [0 1 2]


**STEP 2:** Making **`X`** (Input) dataset and excluding the 5th 'variety' column (our Target).


```python
base_db = db.iloc[:,:-1]
print(base_db.head())
```

       sepal.length  sepal.width  petal.length  petal.width
    0           5.1          3.5           1.4          0.2
    1           4.9          3.0           1.4          0.2
    2           4.7          3.2           1.3          0.2
    3           4.6          3.1           1.5          0.2
    4           5.0          3.6           1.4          0.2


**STEP 3:** Building **`y`** (Target) dataset out of 'variety' column.


```python
target_db = db.iloc[:,-1]
print(target_db.head())
```

    0    0
    1    0
    2    0
    3    0
    4    0
    Name: variety, dtype: int64


**STEP 4:** Loading necessary packages including `Spltr`. Note: for `Spltr` to work properly `torch`, `numpy` and `pandas` shall be installed as well.


```python
import torch
import numpy as np
from spltr import Spltr
```

**STEP 5.1:** Now let's instantiate a Spltr object by including **`X,y`** datasets into it.


```python
splt = Spltr(base_db,target_db)
```

**STEP 5.2:** Alternatively we may load whole datasets and apply basic preprocessing scenarios (exclude columns, change shape and type of the data, etc.). That allows to quickly iterate and find the best approach to current ML task. To demonstrate that, let's reinstantiate `base_db` & `target_db`


```python
base_db = pd.read_csv(link) # Now 'base_db' is a 5-column dataset that includes the 'target' (5th) column.
target_db = link # And 'target_db' is a simple link to the same 5-column dataset.

splt = Spltr(base_db,target_db)
```

Here we start with processing **`X`** by selecting only 4 feature columns and, thus, excluding the target (5th column). Pls note that presented vocabulary-type selection `{column start : column finish}` works only for DataFrames as for now.


```python
splt.process_x(preview=True, usecols={0:4})
```

    [X]: The following DataFrame was loaded:
    
       sepal.length  sepal.width  petal.length  petal.width
    0           5.1          3.5           1.4          0.2
    1           4.9          3.0           1.4          0.2
    
    [X]: Tensor of shape torch.Size([150, 4]) was created


And continue with processing **`y`** by using just the 5th column named 'variety'. Note that CSV columns may be selected by [Int] or [String] as per official pd.read_csv documentation.


```python
splt.process_y(preview=True, usecols=['variety'])
```

    [y]: The following CSV was loaded:
    
       variety
    0        0
    1        0
    
    [y]: Tensor of shape torch.Size([150, 1]) was created



```python
splt.reshape_xy(y_shape=-1) # Reshaping 'y' to be easily used in a classification task
print(splt.shape_y_)
```

    torch.Size([150])


**STEP 6:** Splitting the dataset into train - 30%, test - 20%, which makes validation set equal to 50% (calculated automatically: val = 100% - train - test), and initializing permutation of the data. 

`Spltr.split_data` method may use native `DataLoader` methods like `num_workers`, `pin_memory`, `worker_init_fn`, `multiprocessing_context` and others as per PyTorch documentation.


```python
splt.split_data(splits=(0.3,0.2), perm_seed=3, batch_size=1, num_workers = 1, shuffle=True)
```

    [X,y]: The Data is splitted into 3 datasets of length: Train 45, Test 30, Validation 75.


**STEP 7:** Now let's clean unnecessary variables saved in the memory. This step may be especially useful if you are dealing with a huge datasets and don't want for `X,y tensors` to share memory with `X,y DataLoader objects`.


```python
splt.clean_space()
```

    All variables are deleted. Only Train-Test (Validation) sets are left.


**STEP 8:** Setting up a very simple neural network. Pls mind that the presented network architecture is comprised only to demonstrate how `Spltr` may be adopted. That's not an optimal way to solve Iris classification problem.


```python
from torch import nn, optim
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.input = nn.Linear(4,8)
        self.output = nn.Linear(8,3)
        
    def forward(self,x):
        
        x = F.softsign(self.input(x))
        x = self.output(x)
        return x

model = Net()
criterion = nn.CrossEntropyLoss(reduction='none')
optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.9, eps=1e-08)

for n in range(7):
    train_loss = 0.0
    
    # Fitting a dataset for Training
    
    for train_data, train_target in splt.xy_train:
        model.zero_grad()
        train_result = model(train_data.float())
        loss = criterion(train_result, train_target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    test_correct = 0
    with torch.no_grad():
        
        # Fitting a dataset for Testing
        
        for test_data, test_target in splt.xy_test:
            test_result = model(test_data.float())
            _, predicted = torch.max(test_result, 1)
            test_correct += (predicted == test_target).sum().item()
            
    print(f'Epoch {n+1}. Train loss: ', round(train_loss/len(splt.xy_train), 5))
    print(f'Testing: Correctly identified samples {test_correct} out of {len(splt.xy_test)}', end='\n\n')
    
val_correct = 0

with torch.no_grad():
    
    # Fitting a dataset for Validation
    
    for val_data, val_target in splt.xy_val:
        val_result = model(val_data.float())
        _, val_predicted = torch.max(val_result, 1)
        val_correct += (val_predicted == val_target).sum().item()
            
print(f'VALIDATION: Correctly identified samples {val_correct} out of {len(splt.xy_val)}', end='\n\n')
```

    Epoch 1. Train loss:  1.09528
    Testing: Correctly identified samples 21 out of 30
    
    Epoch 2. Train loss:  0.80778
    Testing: Correctly identified samples 21 out of 30
    
    Epoch 3. Train loss:  0.58353
    Testing: Correctly identified samples 24 out of 30
    
    Epoch 4. Train loss:  0.48647
    Testing: Correctly identified samples 26 out of 30
    
    Epoch 5. Train loss:  0.43741
    Testing: Correctly identified samples 21 out of 30
    
    Epoch 6. Train loss:  0.41516
    Testing: Correctly identified samples 27 out of 30
    
    Epoch 7. Train loss:  0.38361
    Testing: Correctly identified samples 30 out of 30
    
    VALIDATION: Correctly identified samples 73 out of 75
    

