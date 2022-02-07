#!/usr/bin/env python
# coding: utf-8

# In[90]:


import json
import pandas as pd
from pprint import pprint
import tensorflow as tf


# In[46]:


with open('NIKL_DIALOGUE_2020_v1.2/SDRW2000000001.json', 'r', encoding='UTF8') as file:
    data = json.load(file)
    pprint(data)


# In[53]:


# key값들 ?
for item in data:
    print(item)


# In[54]:


data['document'] # 대화 부분


# In[63]:


temp_data = dict()
temp_data = data['document']


# In[69]:


temp_data


# In[67]:


temp_data[0]['id']


# In[80]:


temp_data[0]


# In[98]:


# 대화문만 뽑기
for text in temp_data[0]['utterance']:
    print(text['speaker_id'], ":", text['form'])


# In[ ]:





# In[99]:


from tokenizers import BertWordPieceTokenizer


# In[103]:


tokenizer = BertWordPieceTokenizer(
vocab = None,
clean_text=True,
handle_chinese_chars=True,
strip_accents=False,
lowercase=False,
wordpieces_prefix="##")

tokenizer.train(
files=paths,
limit_alphabet=3000,
vocab_size = args.vocab_size,
min_frequency=50,
show_progress=True)


# In[ ]:





# In[ ]:





# ### 한국어 형태소 분석기

# In[1]:


from konlpy.tag import Kkma, Komoran, Okt, Mecab


# In[2]:


#mec = Mecab()
okt = Okt()
kkm = Kkma()
kom = Komoran()


# In[ ]:





# In[ ]:




