# -*- coding: utf-8 -*-
"""
Created on Tue May 15 23:10:57 2018

@author: Sun
"""

import numpy as np
import pandas as pd
#数据预处理
zb = pd.read_csv(r'C:\Users\Sun\Desktop\zhibiao.csv',header=None)   #读数据.csv格式
v = zb.loc[:,1].values    #等评价事物p的值
j = zb.loc[:,2:].values   #节域
vip1=np.min(j,axis=1)     #节域左侧边界
vip2=np.max(j,axis=1)     #节域右侧边界
col = np.int(j.shape[1]/2)    #经典域个数
row = np.int(j.shape[0])  #评价指标个数
r = [] #初始化新数组r，存储经典域数据
#读入.csv内矩阵数据并做拉直处理,存储为列表
for i in range(1,col+1):
    d = zb.loc[:,i*2:i*2+1].values.T 
    d = d.ravel()
    for j in range(len(d)):
        r.append(d[j])
r = np.array([r]) #列表转一维数组
r = r.reshape(col,2,row) #将一维数组重塑为[经典域左侧，经典域右侧，指标数]的3维数组
aij = np.array([r[i][0][j] for i in range(col) for j in range(row)]) #计算aij
aij = aij.reshape(col,row)
bij = np.array([r[i][1][j] for i in range(col) for j in range(row)]) #计算bij
bij = bij.reshape(col,row)


#计算rij和rijmax

#计算m、n
m = [] #m=(aij+bij)/2
n = [] #n=bij-aij
for i in range(col):
    for j in range(row):
        mr = (r[i][0][j]+r[i][1][j])/2
        nr = r[i][1][j]-r[i][0][j]
        m.append(mr)
        n.append(nr)
m = np.array(m)
n = np.array(n)
m = m.reshape(col,row)
n = n.reshape(col,row)

#计算rij
rij = [] 
for i in range(col):    
    for j in range(row):
        if v[j] <= m[i][j]:
            k = 2*(v[j] - aij[i][j])/n[i][j]
        else: 
            k = 2*(bij[i][j]-v[j])/n[i][j]
        rij.append(k)
rij = np.array(rij)
rij = rij.reshape(col,row)
rij = rij.T
rijmax = np.array([rij[i].max() for i in range(row)]) #计算rijmax


#计算jmax
a1 = []
for t in range(col):
    for s in range(row):
        if v[s] <= bij[t][s] and v[s] > aij[t][s]:
            l = 1
        else: 
            l = 0
        a1.append(l)
a1 = np.array(a1)
a1 = a1.reshape(col,row)
a = np.array([a1[i]*(i+1) for i in range(col)])
jmax = np.sum(a,axis=0)


#计算ri 指标i的数据落入的类别越大，该指标应赋予越大的权重,选上方公式；
#如果指标i的数据落入的类别越大，该指标应赋予越小的权重，选下方公式。
#ri = np.array([jmax[i]*(1+rijmax[i]) for i in range(row)])
ri = np.array([(col-jmax[i]+1)*(1+rijmax[i]) for i in range(row)])

#权重系数w计算
w = np.array([ri[i]/ri.sum() for i in range(row)])

#关联度K计算
vij1 = bij - aij #计算b-a
vij2 = aij + bij #计算a+b
p = np.abs(v - vij2/2)-vij1/2 
pvip = np.abs(v - (vip1+vip2)/2)-(vip2-vip1)/2

K = []
for i in range(col):
    for j in range(row):
        if round(pvip[j],4) != round(p[i][j],4):
            Kjvi = p[i][j]/(pvip[j]-p[i][j])
        else:
            Kjvi = -1*p[i][j]
        K.append(Kjvi)

K = np.array(K)
Kjv = K.reshape(col,row)
Kjp = np.dot(Kjv,w)


#等级评价
from sklearn.preprocessing import minmax_scale
Kjp_ = minmax_scale(Kjp)
j0 = np.argmax(Kjp)+1
jsum = Kjp_.sum()
j_star = np.dot(np.arange(1,col+1),Kjp_)/Kjp_.sum()


#输出结果
print("权重系数为",np.round(w.tolist(),3))
print("权重系数的和为",round(w.sum(),1))
print("待评价事物p关于等级j的关联度 Kj(p)",np.round(Kjp,3))
print("评定事物p的等级 j0=",j0)
print("待评价事物的级别变量特征值 j*=",round(j_star,2))