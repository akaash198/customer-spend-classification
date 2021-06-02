#!/usr/bin/env python
# coding: utf-8

# # IMPORT LIBRARIES

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


# # READING DATA

# In[2]:


data=pd.read_csv("Cust_Spend_Data_New.csv")
data.head(10)


# In[3]:


data.isnull().sum()


# In[4]:


sns.heatmap(data.isnull())


# In[5]:


data.describe()


# In[6]:


data.info()


# In[7]:


sns.pairplot(data);


# In[8]:


data.hist(figsize=(20,20))

plt.show()


# In[9]:


cor = data.corr()



mask = np.zeros_like(cor)

mask[np.triu_indices_from(mask)] = True



plt.figure(figsize=(12,10))



with sns.axes_style("white"):

    sns.heatmap(cor,annot=True,linewidth=2,

                mask = mask,cmap="magma")

plt.title("Correlation between variables")

plt.show()


# In[10]:


sns.countplot(x="FnV_Items",data=data)


# In[11]:


sns.countplot(x="No_Of_Visits",data=data)


# In[12]:


sns.countplot(x="Apparel_Items",data=data)


# In[13]:


sns.countplot(x="Staples_Items",data=data)


# In[14]:


data.groupby('No_Of_Visits')['Avg_Mthly_Spend'].mean().plot(kind='barh', rot=45, fontsize=10, figsize=(15, 8))


# In[15]:


data.groupby('Apparel_Items')['Avg_Mthly_Spend'].mean().plot(kind='barh', rot=45, fontsize=10, figsize=(15, 8))


# In[16]:


data.groupby('FnV_Items')['Avg_Mthly_Spend'].mean().plot(kind='barh', rot=45, fontsize=10, figsize=(15, 8))


# In[17]:


data.groupby('Staples_Items')['Avg_Mthly_Spend'].mean().plot(kind='barh', rot=45, fontsize=10, figsize=(15, 8))


# In[18]:


data.groupby('No_Of_Visits')[ 'Avg_Mthly_Spend'].mean().reset_index()


# In[19]:


data.groupby('Apparel_Items')[ 'Avg_Mthly_Spend','No_Of_Visits'].mean().reset_index()


# In[20]:


data.groupby('FnV_Items')[ 'Avg_Mthly_Spend','No_Of_Visits'].mean().reset_index()


# In[21]:


data.groupby('Staples_Items')[ 'Avg_Mthly_Spend','No_Of_Visits'].mean().reset_index()


# In[22]:


#Count and group by category
category = data.groupby('Avg_Mthly_Spend').agg({'Cust_ID':'count'}).rename(columns={'Cust_ID':'Cust_ID'}).reset_index()
#Get 10 first categories
category2 = category.sort_values(by=['Cust_ID'], ascending = False).head(10)
category2.head()


# # pattern finding

# In[23]:


import plotly.express as px
import apyori
from apyori import apriori


# In[24]:


print("Top 10 frequently sold products(Tabular Representation)")
x = data['Apparel_Items'].value_counts().sort_values(ascending=False)[:10]
fig = px.bar(x= x.index, y= x.values)
fig.update_layout(title_text= "Top 10 frequently sold products (Graphical Representation)", xaxis_title= "Products", yaxis_title="Count")
fig.show()


# In[25]:


print("Top 10 frequently sold products(Tabular Representation)")
x = data['FnV_Items'].value_counts().sort_values(ascending=False)[:10]
fig = px.bar(x= x.index, y= x.values)
fig.update_layout(title_text= "Top 10 frequently sold products (Graphical Representation)", xaxis_title= "Products", yaxis_title="Count")
fig.show()


# In[26]:


print("Top 10 frequently sold products(Tabular Representation)")
x = data['Staples_Items'].value_counts().sort_values(ascending=False)[:10]
fig = px.bar(x= x.index, y= x.values)
fig.update_layout(title_text= "Top 10 frequently sold products (Graphical Representation)", xaxis_title= "Products", yaxis_title="Count")
fig.show()


# In[27]:


transactions = []
for i in range(0, 829):
    transactions.append([str(data.values[i,j]) for j in range(0, 7)])
rules = apriori(transactions, min_support = 0.00030, min_confidence = 0.05, min_lift = 3, max_length = 2, target = "rules")
association_results = list(rules)
print(association_results[0])


# In[28]:


for item in association_results:
    
    pair = item[0]
    items = [x for x in pair]
    
    print("Rule : ", items[0], " -> " + items[1])
    print("Support : ", str(item[1]))
    print("Confidence : ",str(item[2][0][2]))
    print("Lift : ", str(item[2][0][3]))
    
    print("=============================") 

