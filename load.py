#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from imports import *

def dataframe_loader(file_name, data_path = 'Data/'):
    """
    load dataset of text data
    """
    with open(os.path.join(data_path, file_name), "r", encoding = "utf-8") as datafile:
        csv_read = pd.read_csv(datafile)
    print("Dataset loaded")
        
    return csv_read