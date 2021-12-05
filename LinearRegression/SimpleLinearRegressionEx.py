#!/usr/bin/env python
# coding: utf-8

# # 単線形回帰　問題の解答

# 不動産に関するデータセットがあります
# 
# 不動産には値段と床面積などに因果関係があることが一般的です
# 
# データは以下のファイルとして保存してあります
# 'real_estate_price_size.csv'. 
# 
# ここで、単線形回帰を作成してみましょう
# 
# この問題では、従属変数がpriceで独立変数がsizeとなります
# 

# ## ライブラリのインポート

# In[2]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm


# ## データの読み込み

# In[3]:


df = pd.read_csv("./real_estate_price_size.csv")


# In[4]:


df.describe()


# In[ ]:





# ## 回帰の作成

# ### 従属変数と独立変数の定義

# In[5]:


y=df['price']
x=df['size']


# ### データの確認

# In[6]:


plt.scatter(x,y)


# ### 回帰の実行

# In[7]:


x1=sm.add_constant(x)


# In[8]:


results = sm.OLS(y,x1).fit()


# In[9]:


results.summary()


# In[14]:


yhat=223.1787*x1+1.019E5
plt.plot(x1,yhat)
plt.scatter(x,y)


# In[ ]:




