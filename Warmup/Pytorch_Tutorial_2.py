#!/usr/bin/env python
# coding: utf-8

# # **Pytorch Tutorial 2**
# Video: https://youtu.be/VbqNn20FoHM

# In[ ]:


import torch


# **1. Pytorch Documentation Explanation with torch.max**
# 
# 

# In[ ]:


x = torch.randn(4,5)
y = torch.randn(4,5)
z = torch.randn(4,5)
print(x)
print(y)
print(z)


# In[ ]:


# 1. max of entire tensor (torch.max(input) → Tensor)
m = torch.max(x)
print(m)


# In[ ]:


# 2. max along a dimension (torch.max(input, dim, keepdim=False, *, out=None) → (Tensor, LongTensor))
m, idx = torch.max(x,0)
print(m)
print(idx)


# In[ ]:


# 2-2
m, idx = torch.max(input=x,dim=0)
print(m)
print(idx)


# In[ ]:


# 2-3
m, idx = torch.max(x,0,False)
print(m)
print(idx)


# In[ ]:


# 2-4
m, idx = torch.max(x,dim=0,keepdim=True)
print(m)
print(idx)


# In[ ]:


# 2-5
p = (m,idx)
torch.max(x,0,False,out=p)
print(p[0])
print(p[1])


# In[ ]:


# 2-6
p = (m,idx)
torch.max(x,0,False,p)
print(p[0])
print(p[1])


# In[ ]:


# 2-7
m, idx = torch.max(x,True)


# In[ ]:


# 3. max(choose max) operators on two tensors (torch.max(input, other, *, out=None) → Tensor)
t = torch.max(x,y)
print(t)


# **2. Common errors**
# 
# 

# The following code blocks show some common errors while using the torch library. First, execute the code with error, and then execute the next code block to fix the error. You need to change the runtime to GPU.
# 

# In[ ]:


import torch


# In[ ]:


# 1. different device error
model = torch.nn.Linear(5,1).to("cuda:0")
x = torch.randn(5).to("cpu")
y = model(x)


# In[ ]:


# 1. different device error (fixed)
x = torch.randn(5).to("cuda:0")
y = model(x)
print(y.shape)


# In[ ]:


# 2. mismatched dimensions error 1
x = torch.randn(4,5)
y = torch.randn(5,4)
z = x + y


# In[ ]:


# 2. mismatched dimensions error 1 (fixed by transpose)
y = y.transpose(0,1)
z = x + y
print(z.shape)


# In[ ]:


# 3. cuda out of memory error
import torch
import torchvision.models as models
resnet18 = models.resnet18().to("cuda:0") # Neural Networks for Image Recognition
data = torch.randn(2048,3,244,244) # Create fake data (512 images)
out = resnet18(data.to("cuda:0")) # Use Data as Input and Feed to Model
print(out.shape)


# In[ ]:


# 3. cuda out of memory error (fixed, but it might take some time to execute)
for d in data:
  out = resnet18(d.to("cuda:0").unsqueeze(0))
print(out.shape)


# In[ ]:


# 4. mismatched tensor type
import torch.nn as nn
L = nn.CrossEntropyLoss()
outs = torch.randn(5,5)
labels = torch.Tensor([1,2,3,4,0])
lossval = L(outs,labels) # Calculate CrossEntropyLoss between outs and labels


# In[ ]:


# 4. mismatched tensor type (fixed)
labels = labels.long()
lossval = L(outs,labels)
print(lossval)


# **3. More on dataset and dataloader**
# 

# A dataset is a cluster of data in a organized way. A dataloader is a loader which can iterate through the data set.

# Let a dataset be the English alphabets "abcdefghijklmnopqrstuvwxyz"

# In[ ]:


dataset = "abcdefghijklmnopqrstuvwxyz"


# A simple dataloader could be implemented with the python code "for"

# In[ ]:


for datapoint in dataset:
  print(datapoint)


# When using the dataloader, we often like to shuffle the data. This is where torch.utils.data.DataLoader comes in handy. If each data is an index (0,1,2...) from the view of torch.utils.data.DataLoader, shuffling can simply be done by shuffling an index array. 
# 
# 

# torch.utils.data.DataLoader will need two imformation to fulfill its role. First, it needs to know the length of the data. Second, once torch.utils.data.DataLoader outputs the index of the shuffling results, the dataset needs to return the corresponding data.

# Therefore, torch.utils.data.Dataset provides the imformation by two functions, `__len__()` and `__getitem__()` to support torch.utils.data.Dataloader

# In[ ]:


import torch
import torch.utils.data 
class ExampleDataset(torch.utils.data.Dataset):
  def __init__(self):
    self.data = "abcdefghijklmnopqrstuvwxyz"
  
  def __getitem__(self,idx): # if the index is idx, what will be the data?
    return self.data[idx]
  
  def __len__(self): # What is the length of the dataset
    return len(self.data)

dataset1 = ExampleDataset() # create the dataset
dataloader = torch.utils.data.DataLoader(
                        dataset = dataset1, 
                        shuffle = True, 
                        batch_size = 1
              )
for datapoint in dataloader:
  print(datapoint)


# A simple data augmentation technique can be done by changing the code in `__len__()` and `__getitem__()`. Suppose we want to double the length of the dataset by adding in the uppercase letters, using only the lowercase dataset, you can change the dataset to the following.

# In[ ]:


import torch.utils.data 
class ExampleDataset(torch.utils.data.Dataset):
  def __init__(self):
    self.data = "abcdefghijklmnopqrstuvwxyz"
  
  def __getitem__(self,idx): # if the index is idx, what will be the data?
    if idx >= len(self.data): # if the index >= 26, return upper case letter
      return self.data[idx%26].upper()
    else: # if the index < 26, return lower case, return lower case letter
      return self.data[idx]
  
  def __len__(self): # What is the length of the dataset
    return 2 * len(self.data) # The length is now twice as large

dataset1 = ExampleDataset() # create the dataset
dataloader = torch.utils.data.DataLoader(dataset = dataset1,shuffle = True,batch_size = 1)
for datapoint in dataloader:
  print(datapoint)

