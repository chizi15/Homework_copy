#!/usr/bin/env python
# coding: utf-8

# # **Google Colab Tutorial**
# 
# Video: https://youtu.be/YmPF0jrWn6Y
# Should you have any question, contact TAs via <br/> ntu-ml-2022spring-ta@googlegroups.com
# 

# <p><img alt="Colaboratory logo" height="45px" src="/img/colab_favicon.ico" align="left" hspace="10px" vspace="0px"></p>
# 
# <h1>What is Colaboratory?</h1>
# 
# Colaboratory, or "Colab" for short, allows you to write and execute Python in your browser, with 
# - Zero configuration required
# - Free access to GPUs
# - Easy sharing
# 
# Whether you're a **student**, a **data scientist** or an **AI researcher**, Colab can make your work easier. Watch [Introduction to Colab](https://www.youtube.com/watch?v=inN8seMm7UI) to learn more, or just get started below!
# 
# You can type python code in the code block, or use a leading exclamation mark ! to change the code block to bash environment to execute linux code.

# To utilize the free GPU provided by google, click on "Runtime"(執行階段) -> "Change Runtime Type"(變更執行階段類型). There are three options under "Hardward Accelerator"(硬體加速器), select "GPU". 
# * Doing this will restart the session, so make sure you change to the desired runtime before executing any code.
# 

# In[ ]:


import torch
torch.cuda.is_available() # is GPU available
# Outputs True if running with GPU


# In[ ]:


# check allocated GPU type
get_ipython().system('nvidia-smi')


# **1. Download Files via google drive**
# 
#   A file stored in Google Drive has the following sharing link：
# 
# https://drive.google.com/open?id=1sUr1x-GhJ_80vIGzVGEqFUSDYfwV50YW
#   
#   The random string after "open?id=" is the **file_id** <br />
# ![](https://i.imgur.com/77AeV88l.png)
# 
#   It is possible to download the file via Colab knowing the **file_id**, using the following command.
# 
# 
# 
# 
# 

# In[ ]:


# Download the file with file_id "sUr1x-GhJ_80vIGzVGEqFUSDYfwV50YW", and rename it to pikachu.png
get_ipython().system("gdown --id '1sUr1x-GhJ_80vIGzVGEqFUSDYfwV50YW' --output pikachu.png")


# In[ ]:


# List all the files under the working directory
get_ipython().system('ls')


# Exclamation mark (!) starts a new shell, does the operations, and then kills that shell, while percentage (%) affects the process associated with the notebook

# It can be seen that `pikachu.png` is saved the the current working directory. 
# 
# ![](https://i.imgur.com/bonrOlgm.png)
# 
# The working space is temporary, once you close the browser, the files will be gone.
# 

# Double click to view image
# 
# ![](https://i.imgur.com/DTywPzAm.png)

# **2. Mounting Google Drive**
# 
#   One advantage of using google colab is that connection with other google services such as Google Drive is simple. By mounting google drive, the working files can be stored permanantly. After executing the following code block, your google drive will be mounted at `/content/drive`

# ![](https://i.imgur.com/IbMf5Tg.png)

# In[ ]:


try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    drive.mount('/content/drive')
else:
    print("Not running in Google Colab")
    # 在本地环境中执行其他操作


from google.colab import drive
drive.mount('/content/drive')


# After mounting the drive, the content of the google drive will be mounted on a directory named `MyDrive`
# 
# ![](https://i.imgur.com/jDtI10Cm.png)

# After mounting the drive, all the changes will be synced with the google drive.
# Since models could be quite large, make sure that your google drive has enough space.

# In[ ]:


get_ipython().run_line_magic('cd', '/content/drive/MyDrive')
#change directory to google drive
get_ipython().system('mkdir ML2022 #make a directory named ML2022')
get_ipython().run_line_magic('cd', './ML2022')
#change directory to ML2022


# Use bash command pwd to output the current directory

# In[ ]:


get_ipython().system('pwd #output the current directory')


# Repeat the downloading process, this time, the file will be stored permanently in your google drive.

# In[ ]:


get_ipython().system("gdown --id '1sUr1x-GhJ_80vIGzVGEqFUSDYfwV50YW' --output pikachu.png")


# Check the file structure
# 
# ![](https://i.imgur.com/DbligmOt.png)

# For all the homeworks, the data can be downloaded and stored similar as demonstrated in this notebook. 
