from os import read
import random
import math
#from tsp_file import read
from scipy.spatial import distance

def init_sol2(d,nodelist):
    i,j=1,0
    t=[]
    j=random.randint(0,d-1)
    t.append(j)
    while i < d:
        #j=minx(j,nodelist)
        j=minxx(j,nodelist,t)
        t.append(j)
        i=i+1
    return t

def minxx(i,nodelist,t):
    min=200000
    mini=0
    if i == 0 and t.__contains__(0)==False:
      min=nodelist[i][1]
      mini=1
    else:
        if t.__contains__(i)==False:
          min=nodelist[i][0]
          mini=0
    for j in range(0, len(nodelist)):
        if i!=j and t.__contains__(j)==False:
            if min >nodelist[i][j] :
                 min=nodelist[i][j] 
                 mini=j
                
    return mini 



'''
def f_c1(traget,n):
    s=0.0
    i=0
    if traget!=[]:
     if len(traget)==1:
        s=traget[0]
     else:
      for i in range (0,n-1):
        s=s+ nodelist[traget[i]][traget[i+1]]
      s=s+ nodelist[traget[-1]][traget[0]]
    return s
'''
#nodelist=read('berlin52.tsp')
#print(f_c1(init_sol2(52,nodelist),52))