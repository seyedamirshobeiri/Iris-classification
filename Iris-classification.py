#!/usr/bin/env python
# coding: utf-8

# In[113]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[114]:


iris=load_iris()


# In[115]:


x_train,y_train,x_test,y_test=train_test_split(iris.data,iris.target)


# In[116]:


model=LogisticRegression(random_state=1,solver='newton-cg',multi_class='auto')


# In[117]:


model.fit(x_train,x_test)


# In[118]:


predict=model.predict(y_train)


# In[119]:


acc=accuracy_score(predict,y_test)
txt = "The accuracy is {:.0f}% ."
print(txt.format(acc*100))


# In[ ]:





# In[ ]:




