
# What is it?

`Spltr` is a simple PyTorch-based data loader and splitter.
It may be used to load i) arrays and ii) matrices or iii) Pandas DataFrames 
and iv) CSV files containing numerical data with the subsequent split it into 
train, test (validation) subsets in the form of PyTorch DataLoader objects.
The special emphesis was given to ease of usage and automation of many
data-handling procedures.

Originally it was developed in order to speed up a data preparation stage
for number of trivial ML tasks. Hope it may be useful for you as well.

# Main Features

`Spltr.process_x|y` : converts loaded Input/Target data into PyTorch Tensors 
    with ability to i) preview, ii) define tensor dtype, iii) set the desired 
    device of returned tensor (CPU/GPU), iv) use selected rows/columns from 
    Input/Target data sources or a single data table for processing 
    (for CSV and Pandas DataFrame only).
           
`Spltr.reshape_xy` : reshapes subsets.
    
`Spltr.split_data` : splits data subsets into train, test (validation) 
    PyTorch DataLoader objects.
    
`Spltr.clean_space` : optimizes mamory by deleting unnecessary variables.

# Installation

pip install spltr

# License

OSI Approved :: MIT License

# Documentation

https://github.com/maksymsur/Spltr

# Dependencies

+ torch >= 1.1.0

+ numpy >= 1.16.4

+ pandas >= 0.24.2

# Example of usage

Hereunder we'll build a simple neural network and describe how the `Spltr` may be used in the process.


But first, let's import all the necessary modules:


```python
import numpy as np
import pandas as pd
import torch
from torch.utils import data
from torch import nn, optim
import torch.nn.functional as F
```

Loading and reading an Iris dataset to be used as an example. The dataset may be found at: https://github.com/maksymsur/Spltr/blob/master/dataset/iris_num.csv


```python
db = pd.read_csv('/home/your_path_here/iris_num.csv')
print(db.head())
```

       sepal.length  sepal.width  petal.length  petal.width  variety
    0           5.1          3.5           1.4          0.2        0
    1           4.9          3.0           1.4          0.2        0
    2           4.7          3.2           1.3          0.2        0
    3           4.6          3.1           1.5          0.2        0
    4           5.0          3.6           1.4          0.2        0


By building the network we'll try to predict the 'variety' column. This column identifies Iris species: Setosa = 0, Versicolor = 1 and Virginica = 2. Let's verify that it is comprised from the mentioned 3 unique values.


```python
print(db.variety.unique())
```

    [0 1 2]


Making **`X`** (Input) dataset excluding the 'variety' column.


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


Making **`y`** (Target) dataset including the 'variety' column and another column 'petal.width'. The latter will be excluded by `Spltr` later just to demonstrate how easily we may modify loaded datasets.


```python
target_db = db.iloc[:,-2:]
print(target_db.head())
```

       petal.width  variety
    0          0.2        0
    1          0.2        0
    2          0.2        0
    3          0.2        0
    4          0.2        0


Loading **`X,y`** datasets into `Spltr`.


```python
from spltr import Spltr
splt = Spltr(base_db,target_db)
```

Preprocessing **`X`** (Input data) by converting it into PyTorch Tensors.


```python
splt.process_x(preview=True)
```

    [X]: The following DataFrame was loaded:
    
       sepal.length  sepal.width  petal.length  petal.width
    0           5.1          3.5           1.4          0.2
    1           4.9          3.0           1.4          0.2
    
    [X]: Tensor of shape torch.Size([150, 4]) was created


Preprocessing **`y`** (Target dataset) by converting it into PyTorch Tensors as well. 

At this stage we also indicate that only 2nd column from the `y` dataset will be used. Thus, by specifying `usecols = 1` ('variety' index) we exclude the 'petal.width' column with index 0. Note, `usecols` may also work with strings and expression `usecols = 'variety'` is also viable.


```python
splt.process_y(preview=True, usecols=1)
```

    [y]: The following DataFrame was loaded:
    
    0    0
    1    0
    Name: variety, dtype: int64
    
    [y]: Tensor of shape torch.Size([150]) was created


Splitting the dataset into train - 30% and test - 20% which makes validation set equal to 50% (calculated automatically: val = 100% - train - test) and initializing permutation of the data.


```python
splt.split_data(splits=(0.3,0.2), perm_seed=8)
```

    [X,y]: The Data is splited into 3 datasets of length: Train 45, Test 30, Validation 75.
           DataLoaders with Batch Size 1 and Shufle False are created


Setting up a very simple neural network. Pls note that the network architecture is comprised only to demonstrate how `Spltr` may be adopted. That's not an optimal way to solve Iris classification problem. 


```python
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
optimizer = optim.RMSprop(model.parameters())

for n in range(10):
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

    Epoch 1. Train loss:  1.01279
    Testing: Correctly identified samples 14 out of 30
    
    Epoch 2. Train loss:  0.57477
    Testing: Correctly identified samples 14 out of 30
    
    Epoch 3. Train loss:  0.48212
    Testing: Correctly identified samples 14 out of 30
    
    Epoch 4. Train loss:  0.44074
    Testing: Correctly identified samples 16 out of 30
    
    Epoch 5. Train loss:  0.41144
    Testing: Correctly identified samples 24 out of 30
    
    Epoch 6. Train loss:  0.38251
    Testing: Correctly identified samples 29 out of 30
    
    Epoch 7. Train loss:  0.34853
    Testing: Correctly identified samples 29 out of 30
    
    Epoch 8. Train loss:  0.31831
    Testing: Correctly identified samples 29 out of 30
    
    Epoch 9. Train loss:  0.29182
    Testing: Correctly identified samples 29 out of 30
    
    Epoch 10. Train loss:  0.27387
    Testing: Correctly identified samples 30 out of 30
    
    VALIDATION: Correctly identified samples 74 out of 75
    


As you might have noticed, we obtained pretty good results even by using a very few samples (45) for training the network.
