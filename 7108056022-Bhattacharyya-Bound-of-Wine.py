
# coding: utf-8

# In[1]:


import numpy as np
from numpy import *
from numpy.linalg import  *
import math


# # 讀取資料轉成numpy裡的array(資料前處理)

# ## 在創建另一個資料檔，沒有第一排類別，用來算變異數

# In[2]:


data = []
with open("C:/Users/qscf6/Desktop/wine.data", "r") as f:
    lines = f.readlines()
    for line in lines:
        a = line.split(",")
        a[-1] = a[-1].split("\n")[0]
        a = list(map(float , a))
        data.append(np.array(a))
covdata = []
with open("C:/Users/qscf6/Desktop/winedata.data", "r") as f:
    lines = f.readlines()
    for line in lines:
        a = line.split(",")
        a[-1] = a[-1].split("\n")[0]
        a = list(map(float , a))
        covdata.append(np.array(a))


# # 統計每個類別有幾筆資料，進而幫助我們算出平均數和變異數，算出先驗機率

# In[10]:


length = len(data)
count1 = 0
count2 = 0
count3 = 0
for i in range (length):
    if data[i][0] == 1:
        count1 = count1 + 1
    elif data[i][0] == 2:
        count2 = count2 + 1
    else:
        count3 = count3 + 1
print(count1)
P1 = count1 / length
print(P1)
print(count2)
P2 = count2 / length
print(P2)
print(count3)
P3 = count3 / length
print(P3)


# # 算出每一類的平均數與變異數

# # 第一類

# In[4]:


averge1 = []
for j in range(1,14):
    feature = []
    for i in range (0,count1):
        feature.append(data[i][j])
    averge1.append(np.mean(feature))
averge1 = np.array(averge1)
print(averge1)
covdata1 = []
sd1 = []
for i in covdata[:count1]:
    covdata1.append(i)
covdata1 = np.array(covdata1)
covdata1 = covdata1.T
sd1 = np.cov(covdata1)
print(sd1)


# # 第二類

# In[5]:


averge2 = []
for j in range(1,14):
    feature = []
    for i in range (count1,count1+count2):
        feature.append(data[i][j])
    averge2.append(np.mean(feature))
averge2 = np.array(averge2)
print(averge2)
covdata2 = []
sd2 = []
for i in covdata[count1:count1+count2]:
    covdata2.append(i)
covdata2 = np.array(covdata2)
covdata2 = covdata2.T
sd2 = np.cov(covdata2)
print(sd2)


# # 第三類

# In[6]:


averge3 = []
for j in range(1,14):
    feature = []
    for i in range (count1+count2,count1+count2+count3):
        feature.append(data[i][j])
    averge3.append(np.mean(feature))
averge3 = np.array(averge3)
covdata3 = []
sd3 = []
for i in covdata[count1+count2:count1+count2+count3]:
    covdata3.append(i)
covdata3 = np.array(covdata3)
covdata3 = covdata3.T
sd3 = np.cov(covdata3)
print(sd3)


# # 算出兩兩類別之間的錯誤率上限

# # 第一類與第二類

# In[22]:


an12 = float(1/8)*(averge2-averge1).T.dot(np.linalg.inv((sd1+sd2)/2)).dot(averge2-averge1) + float(1/2)*np.log(det((sd1+sd2)/2)/math.sqrt(det(sd1)*det(sd2)))
print(an12)
an12 = math.sqrt(P1*P2)*math.exp(-(an12))
print(an12)


# # 第一類與第三類

# In[21]:


an13 = float(1/8)*(averge3-averge1).T.dot(np.linalg.inv((sd1+sd3)/2)).dot(averge3-averge1) + float(1/2)*np.log(det((sd1+sd3)/2)/math.sqrt(det(sd1)*det(sd3)))
print(an13)
an13 = math.sqrt(P1*P3)*math.exp(-(an13))
print(an13)


# # 第二類與第三類

# In[20]:


an32 = float(1/8)*(averge3-averge2).T.dot(np.linalg.inv((sd2+sd3)/2)).dot(averge3-averge2) + float(1/2)*np.log(det((sd2+sd3)/2)/math.sqrt(det(sd2)*det(sd3)))
print(an32)
an32 = math.sqrt(P2*P3)*math.exp(-(an32))
print(an32)

