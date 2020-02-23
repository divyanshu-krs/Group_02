#!/usr/bin/env python
# coding: utf-8

# In[70]:


import pandas as pd
from gensim.models import FastText
import re
import numpy as np
from libsvm import svmutil
import libsvm


# In[71]:


data = pd.read_excel("NewsHeadlines_01012019_30092019.xlsx")
# ls = [1 if i < 4761 else 0 for i in range(47611)]
# index = data["Like Count"].tolist()
data["Category"].replace({"Tech": "Technology","-":"Uncategorized", "Sports":"Sport"}, inplace=True)
data.head()


# In[4]:


train_df = data["Summary"]


# In[7]:


train_df = train_df.values.tolist()


# In[14]:


summary = []
for sent_str in train_df:
#     sent_str = re.sub(r"'", "", str(sent_str).lower())
    tokens = re.sub(r"[^a-z0-9]+", " ", str(sent_str).lower()).split()
    summary.append(tokens)
summary


# In[3]:


model_summary = FastText(summary , size=100, window=5, min_count=5, workers=4,sg=1)


# In[20]:


model_summary.wv.most_similar("indian")


# In[4]:


# model_summary.save("model_summary.bin")
model_summary = FastText.load("model_summary.bin")


# In[72]:


category = set(data["Category"].values.tolist())


# In[73]:


category


# In[74]:


headlines= []
for sent_str in data["Headline"]:
#     sent_str = re.sub(r"'", "", str(sent_str).lower())
    tokens = re.sub(r"[^a-z0-9]+", " ", str(sent_str).lower()).split()
    if(tokens == []):
        tokens = "nan"
    headlines.append(tokens)


# In[75]:


cat_vec = {}
for c in category:
    cat_vec[c] = model_summary[c]


# In[76]:


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


# head_vec[0]


# In[11]:


#  head_vec[:4261]


# In[36]:


help(svmutil)


# In[126]:


train = head_vec[:int(0.8*(47611))]
res_train = list(np.log2(np.array(data["Like Count"][:int(0.8*47611)].tolist())+1))
test = head_vec[int(0.8*47611):]
res_test = list(np.log2(np.array(data["Like Count"][int(0.8*47611):].tolist())+1))
print(len(train), len(res_train))


# In[ ]:


model_svm_reg = svmutil.svm_train(res_train, train,'-s 4 -v 10 -t 2 -c 2')


# In[78]:


# res_train[31000:31100]


# In[127]:


model_svm = svmutil.svm_train(res_train, train,'-s 4 -t 2 -c 2')


# In[128]:


y_reg = svmutil.svm_predict(res_test+res_train, test+train, model_svm)


# In[148]:


len(y_reg[0])


# In[139]:


svmutil.svm_save_model("model_svm_reg.bin",model_svm)


# In[102]:


from sklearn.metrics import r2_score, mean_absolute_error


# In[143]:


r2 = r2_score(res_test+res_train,y_reg[0])
r2


# In[103]:


# res_test


# In[132]:


mae = mean_absolute_error(res_test+res_train, y_reg[0])
mae


# In[133]:


from scipy.stats import kendalltau
kt = kendalltau(res_test+res_train, y_reg[0])
kt


# In[120]:


y_reg2 = svmutil.svm_predict(res_train, train, model_svm)


# In[134]:


r2 = r2_score(res_train,y_reg2[0])
r2


# In[135]:


mae = mean_absolute_error(res_train, y_reg2[0])
mae


# In[136]:


kt = kendalltau(res_train, y_reg2[0])
kt


# In[137]:


y_reg3 = svmutil.svm_predict(res_test, test, model_svm)


# In[138]:


r2 = r2_score(res_test,y_reg3[0])
r2


# In[140]:


mae = mean_absolute_error(res_test, y_reg3[0])
mae


# In[142]:


from scipy.stats import kendalltau
kt = kendalltau(res_test, y_reg3[0])
kt


# In[ ]:




