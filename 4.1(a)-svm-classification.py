#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from gensim.models import FastText
import re
import numpy as np
from libsvm import svmutil
import libsvm


# In[2]:


data = pd.read_excel("NewsHeadlines_01012019_30092019.xlsx")
ls = [1 if i < 4761 else 0 for i in range(47611)]
# data.set_index("Like Count", inplace=True)
data.sort_values("Like Count", inplace=True,ascending=False)
# print(data.head())
data['class'] = ls
index = data["Like Count"].tolist()
data["Category"].replace({"Tech": "Technology","-":"Uncategorized", "Sports":"Sport"}, inplace=True)
data.head()


# In[3]:


model_summary = FastText.load("model_summary.bin")


# In[6]:


category = set(data["Category"].values.tolist())
headlines= []
for sent_str in data["Headline"]:
#     sent_str = re.sub(r"'", "", str(sent_str).lower())
    tokens = re.sub(r"[^a-z0-9]+", " ", str(sent_str).lower()).split()
    if(tokens == []):
        tokens = "nan"
    headlines.append(tokens)
    
cat_vec = {}
for c in category:
    cat_vec[c] = model_summary[c]    
    
head_vec = []
n = len(headlines)
for h in range(n):
    sum1 = np.array([0.0 for i in range(100)])
    for tok in headlines[h]:
        sum1+=model_summary[tok]
#     print(sum1)
    sum1 = sum1/len(headlines[h])
#     print(sum1)
    sum1 = np.concatenate((sum1,cat_vec[data["Category"][h]]))
    head_vec.append(sum1)


# In[10]:


train = head_vec[:4261]+head_vec[4761:4761*2]
res_train = data["class"][:4261].tolist()+data["class"][4761:4761*2].tolist()
test = head_vec[4261:4761]+head_vec[int(0.8*47611):]
res_test = data["class"][4261:4761].tolist()+data["class"][int(0.8*47611):].tolist()
print(len(train), len(res_train))


# In[8]:


model_svm_5050 = svmutil.svm_train(res_train, train,'-t 2 -c 2')


# In[11]:


y_50 = svmutil.svm_predict(res_test, test, model_svm_5050)


# In[12]:


y_1 = svmutil.svm_predict(res_test[:500], test[:500], model_svm_5050)


# In[15]:


y_1[2]


# In[35]:


from sklearn.metrics import confusion_matrix, auc, roc_curve


# In[21]:


confusion_matrix(res_test, y_50[0], labels=[0,1])


# In[22]:


tn, fp, fn, tp = confusion_matrix(res_test, y_50[0], labels=[0,1]).ravel()


# In[32]:


recall = tp/(tp+fn)
recall


# In[33]:


precision = tp/(tp+fp)
precision


# In[30]:


f1 = precision*recall*2/(precision+recall)
f1


# In[40]:


fpr, tpr, thresholds = roc_curve(res_test, y_50[0])
AUC = auc(fpr, tpr)
AUC


# In[41]:


svmutil.svm_save_model("model_svm.bin",model_svm_5050)


# In[45]:


svmutil.svm_train(res_train, train,'-v 10 -t 2 -c 2')


# In[ ]:





# In[ ]:




