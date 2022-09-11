#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
#Ensure that your current working directory is inside the raw_data folder where all the user data files are located
os.chdir("..\\raw_data")


# In[3]:


import pandas as pd

files = os.listdir('.') #Creates a list of the names of files in the current directory

for file in files:
    if file == files[0]: #If the file is the first file in the list
        df_complete = pd.read_csv(file)
    else:
        df_current = pd.read_csv(file)
        df_complete = pd.concat([df_complete, df_current]) #Concatenate all the csv files into one large dataframe


# In[4]:


df_complete = df_complete.reset_index()
df_complete


# In[5]:


files_without_ext = []

for file in files:
    name = file.split('.')[0]
    files_without_ext.append(name)
    
files_without_ext #Create a list of the user IDs


# In[6]:


UUID = [] #Create a list that is the same length as df_complete
i = 0
for file in files: 
    df = pd.read_csv(file)
    length = len(df)
    for k in range(length):
        UUID.append(files_without_ext[i]) #Append the user ID to the list k number of times where k is the length of the current dataframe
    i+=1 


# In[7]:


df_complete.insert(0, "UUID", UUID)
df_complete #Insert the list of UUID to the dataframe


# In[8]:


df_complete.drop(columns = ["index"], inplace = True)
df_complete #Drop the index column


# In[9]:


df_complete.head()


# In[10]:


df_complete.to_csv("../aggregated_data/aggregated_data.csv", index = False) #Export the CSV file to the current directory (raw_data)

