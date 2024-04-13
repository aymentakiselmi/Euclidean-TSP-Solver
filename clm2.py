from re import L
from tkinter import E
from mpi4py import MPI
from k_means_tsp import main_kmeans
from sol import generation_sols
import numpy as np
import os
from config import *
from scipy.spatial import distance
from copy import deepcopy
from anytree import Node,RenderTree
import anytree
import kppv
import ga_f
import ga2
import random
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AffinityPropagation, KMeans
from scipy.spatial.distance import cdist

nodes=[]
node=[]
root=[]
clusters={}
best={}
fitnesses=[]
is_l=False
g=0
fbs=100000.0
bs=[]
weakness_f=10900.0
pal=0
l4k=[]
zone=False
zf=[]
gln=0
ln=0
pal=0
def read_data(file_location: str):
    if os.path.isfile(file_location):
        with open(file_location) as file: # Use file to refer to the file object
            test_case_count = int(file.readline())
            test_case = list()
            for _ in range(test_case_count):
                city_count = int(file.readline())
                cities = list()
                for count in range(city_count):
                    #line = file.readline().split(',')
                    #x, y = list(map(lambda x: float(x), line))
                    x,y=file.readline().strip().split()[1:]
                    cities.append([float(x),float(y)])
                optimal = float(file.readline())
                
            
        return cities

def ind(X,c):
    cluss=[]
    j=0
    i=0
    while  j< len(c):
        if c[j,0]==X[i,0] and c[j,1]==X[i,1]:
             #print("er",X[i,0],c[j],i,j)
             cluss.append(i)
             j=j+1
             i=0
        else:
            i=i+1
        
    return cluss

def cons_cluster(ds,gds):
    clusters= k_dpc(ds) 
    c1=ind(gds,clusters[0])
    c2=ind(gds,clusters[1])
    c3=ind(gds,clusters[2]) 
    return [c1,c2,c3]

def ctree(ds):
    global root
    nodes.append(Node(''.join('0')))
    clusters['0']=[[i for i in range(0,100)]]
    i=0
    root=nodes[0]
    adr=nodes[0].name
    children=cons_cluster(ds,ds)
    pparent=nodes[0]
    del nodes[0]
    nodes.insert(0,Node(''.join(adr+'3'), parent=pparent))
    nodes.insert(0,Node(''.join(adr+'2'), parent=pparent))
    nodes.insert(0,Node(''.join(adr+'1'), parent=pparent))
    clusters[adr+'1']=children[0]
    clusters[adr+'2']=children[1]
    clusters[adr+'3']=children[2]
    while nodes!=[]:
        adr=nodes[0].name
        pparent=nodes[0]
        if len(clusters[adr])>=4:
                bas=data_value(clusters[adr],ds)
                #print(bas)
                children=cons_cluster(bas,ds)
                del nodes[0]
                nodes.insert(0,Node(''.join(adr+'3'), parent=pparent))
                nodes.insert(0,Node(''.join(adr+'2'), parent=pparent))
                nodes.insert(0,Node(''.join(adr+'1'), parent=pparent))
                clusters[adr+'1']=children[0]
                clusters[adr+'2']=children[1]
                clusters[adr+'3']=children[2]

        else:
            del nodes[0]
        i=i+1

def data_value(fils,X):
    dataset=[]
    
    dataset=np.array([X[j] for j in fils])
    
    return dataset

def select_leafs():
    leafs={}
    keys=clusters.keys()
    for i in nodes:
        if i.is_leaf==True:
             leafs[i.name]=clusters[i.name]
    return leafs

def next_level1(leafs):
    
     next_leafs={}
     key=leafs.keys()
     for i in key:
         c=is_next_level(i,leafs)
         if c<2:
             next_leafs[i]=leafs[i]
         else: 
             if next_leafs.keys().__contains__(i[:-1])==False:    
                        next_leafs[i[:-1]]=agreg(leafs,i)
        
                    

     leafs={}
     leafs=next_leafs
     return leafs

def d_size(g,m,l):
    nk=[]
    for i in range(len(g)):
          if (m.__contains__(g[i])==False and l.__contains__(g[i])==False) or (m.__contains__(g[i])==True and l.__contains__(g[i])==True) or (m.__contains__(g[i])==True and l.__contains__(g[i])==False):
                       nk.append(g[i])
    return nk

def sous_m(traget,n,nodelist):
    s=[]
    ss=[]
    i=0
    if traget!=[]:
     if len(traget)==1:
        s=traget[0]
     else:
      for j in range (0,n):
       for i in range (0,n):
         if i==j:
            s.append(0)
         else:
            #s.append(nodelist[traget[j]][traget[i]])
            s.append(distance.euclidean(nodelist[traget[j]],nodelist[traget[i]]))
       ss.append(s)
       s=[]
    return ss

def bbnew(data,iu):
  exact=[]
  for i in range(0,len(data)):
        exact.append(data[iu[i]])
  data=exact.copy()
  return data

def agreg(leafs,i):
    nb=get_nighbors(i)
    nb.append(i)
    nbs=[]
    for i in range(3):
        nbs=nbs+leafs[nb[i]]
    cond=sous_m(nbs,len(nbs),X)
    path=kppv.init_sol2(len(cond),cond)
    #path=init_sol(len(cond))
    path=bbnew(nbs,path)
    return path

def is_next_level(nb,leafs):
        key=leafs.keys()
        nbs=get_nighbors(nb)
        c=0
        for i in key:
            if i==nbs[0] or i==nbs[1]:
                    c=c+1
        return c

def f_c11(traget,n,nodelist):
    s=0.0
    i=0
    if traget!=[]:
     if len(traget)==1:
        s=traget[0]
     else:
      for i in range (0,n-1):
        s=s+ distance.euclidean(nodelist[traget[i]],nodelist[traget[i+1]])
      s=s+ distance.euclidean(nodelist[traget[-1]],nodelist[traget[0]])
      #s=s+ nodelist[traget[-1]][traget[0]]
    return s

def f_c111(traget,n,nodelist):
    s=0.0
    i=0
    if traget!=[]:
     if len(traget)==1:
        s=traget[0]
     else:
      for i in range (0,n-1):
        s=s+ distance.euclidean(nodelist[traget[i]],nodelist[traget[i+1]])
      #s=s+ distance.euclidean(nodelist[traget[-1]],nodelist[traget[0]])
      #s=s+ nodelist[traget[-1]][traget[0]]
    return s

def get_nighbors(nb):
    nbs=[]
    if nb[-1]=='1':
           s=deepcopy(nb)
           nbs.append(s[:-1]+'2')
           nbs.append(s[:-1]+'3')
    else:
        if nb[-1]=='2':
           s=deepcopy(nb)
           nbs.append(s[:-1]+'1')
           nbs.append(s[:-1]+'3')
        else:
             s=deepcopy(nb)
             nbs.append(s[:-1]+'1')
             nbs.append(s[:-1]+'2')

    return nbs
    
def generate_2_random_index(cities) -> int:
    return random.sample(range(len(cities)), 2)
def calc_fitns(sols,X):
    fi=[]
    for i in sols:
        fi.append(f_c11(i,len(i),X))
    return fi

def update_zf(zf):
    for i in range(len(zf)):
        zf[i]=zf[i][:-1]
    return zf

def length_tr(node):
    l=0
    for i in node: 
        if len(i)>l:
            l=len(i)
    return l

def calc_fitns1(sols,X):
    fi=[]
    for i in sols:
        fi.append(f_c111(i,len(i),X))
    return fi


def bbcast(data):
       if rank == 0:
            comm.send(data, dest=node.index(nodes+'1'), tag=11)
            comm.send(data, dest=node.index(nodes+'2'), tag=11)
            comm.send(data, dest=node.index(nodes+'3'), tag=11)
       else:
               p=comm.recv(source=node.index(nodes[:-1]), tag=11)
               if node.__contains__(nodes+'1')==True:
                 comm.send(p, dest=node.index(nodes+'1'), tag=11)
                 comm.send(p, dest=node.index(nodes+'2'), tag=11)
                 comm.send(p, dest=node.index(nodes+'3'), tag=11)
               return p
       return data

def calculate_similarity_matrix(data):
    # Calculate the similarity matrix using Euclidean distance
    similarity_matrix = -np.sqrt(np.sum((data[:, np.newaxis] - data) ** 2, axis=2))
    return similarity_matrix

def k_affinity_propagation(data, k):
    # Step 1: Perform K-means to obtain initial cluster centers
    kmeans = KMeans(n_clusters=k, random_state=0)
    cluster_labels = kmeans.fit_predict(data)
    cluster_centers = kmeans.cluster_centers_

    # Step 2: Calculate the similarity matrix using Affinity Propagation approach
    similarity_matrix = calculate_similarity_matrix(data)

    # Step 3: Update cluster centers using the Affinity Propagation algorithm
    for _ in range(100):  # Maximum number of iterations
        old_cluster_centers = cluster_centers.copy()

        # Update cluster centers as the most representative points in each cluster
        for cluster_label in range(k):
            cluster_indices = np.where(cluster_labels == cluster_label)[0]
            if len(cluster_indices) > 0:
                similarity_values = similarity_matrix[cluster_indices, :][:, cluster_indices]
                representative_index = np.argmax(np.sum(similarity_values, axis=1))
                cluster_centers[cluster_label] = data[cluster_indices[representative_index]]

        # Assign data points to the closest cluster center
        distances = np.sqrt(np.sum((data[:, np.newaxis] - cluster_centers) ** 2, axis=2))
        cluster_labels = np.argmin(distances, axis=1)

        # Check for convergence
        if np.allclose(cluster_centers, old_cluster_centers):
            break
    clusterss=[] 
    id=[]
    labels=cluster_labels
    centroids = cluster_centers
    colors = ['#4EACC5', '#FF9C34', '#4E9A06',]
    for k, col in zip(range(3), colors):
         my_members = labels == k
         cluster_center = centroids[k]
         clusterss.append(data[my_members])
         id.append(my_members)
     
    return clusterss

def k_dpc(X):
     # Choose number of clusters
    K = 3

    # Compute pairwise distances between all data points
    nbrs = NearestNeighbors(n_neighbors=K+1).fit(X)
    distances, indices = nbrs.kneighbors(X)

    # Set threshold distance (rho) for density peaks
    rho = 0.5

    # Compute local density of each data point
    density = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
         density[i] = 1.0 / (np.max(distances[i, 1:]) + rho)

    # Find K density peaks
    peaks = np.argsort(density)[::-1][:K]

    # Assign each data point to its closest density peak
    labels = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
      if i in peaks:
         labels[i] = np.where(peaks == i)[0][0]
      else:
         dist_to_peaks = cdist(X[i, np.newaxis], X[peaks, :])
         closest_peak = np.argmin(dist_to_peaks)
         labels[i] = closest_peak  
    clusterss=[] 
    clusters1=[] 
    clusters2=[] 
    clusters3=[] 
    id=[]
    for i in range(len(labels)):
        if labels[i]==0: 
              clusters1.append(X[i])
        else:
           if labels[i]==1: 
                  clusters2.append(X[i])
           else:
                  clusters3.append(X[i])
    clusterss.append(np.array(clusters1))
    clusterss.append(np.array(clusters2))
    clusterss.append(np.array(clusters3))
    return clusterss

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size=comm.Get_size()

X=read_data(DATA_FILE)
X=np.array(X)  
if rank==0:
    ctree(X)
    node=[node.name for node in anytree.LevelOrderIter(root)]
    nodes=[node for node in anytree.LevelOrderIter(root)]
    leafs=select_leafs()
    best=next_level1(leafs)
    nodes=d_size(node,best,leafs)
    node=nodes
    mt=sous_m([i for i in range(len(X))],len(X),X)

node = comm.bcast(node, root=0)
nodes = comm.scatter(nodes, root=0)
best = comm.bcast(best, root=0)
key=list(best.keys())
for i in key:
    if i==nodes:
          is_l=True
lf=key
     
if is_l==False and rank==0:
     cod=ga_f.init_pop_not_leaf(5)
     #cod=ga_f.init_pop_not_leaf(512)
     #print('er1',len(cod))
     children1=comm.recv(source=node.index(nodes+'1'), tag=11)
     children2=comm.recv(source=node.index(nodes+'2'), tag=11)
     children3=comm.recv(source=node.index(nodes+'3'), tag=11)
     sols=ga_f.decoding_pop(cod,children1,children2,children3)
     fitnesses=calc_fitns(sols,X)
     l4k=ga2.tour_list(cod,fitnesses)
     '''
     for k in range(0,int(len(cod)*0.5)):
       r =random.randint(0,len(lf)-1)
       zf.append(lf[r])
     '''
     gln=length_tr(node)
     #ln = random.randint(1, gln)
     ln=gln
     
     #weakness_f=ga2.update_w(fitnesses)
     #cod,sols,fitnesses=ga2.kill_invlid_ind(cod,sols,fitnesses,l4k)
     fbs=min(fitnesses)
     print(fbs)
     bs=sols[fitnesses.index(fbs)]
     
else:
 if is_l==False:
     #print(rank,nodes)
     cod=ga_f.init_pop_not_leaf(5)
     #cod=ga_f.init_pop_not_leaf(512)
     #print('er2',len(cod))
     children1=comm.recv(source=node.index(nodes+'1'), tag=11)
     children2=comm.recv(source=node.index(nodes+'2'), tag=11)
     children3=comm.recv(source=node.index(nodes+'3'), tag=11)
     sols=ga_f.decoding_pop(cod,children1,children2,children3)
     comm.send(sols, dest=node.index(nodes[:-1]), tag=11)
 else:
     #print(rank,nodes,best[nodes])
     #print(rank,nodes,best[nodes])
     sols=ga_f.init_pop_leaf(512,best[nodes],X)
     #print('er3',rank,sols)
     comm.send(sols, dest=node.index(nodes[:-1]), tag=11)

fitnesses = bbcast(fitnesses)
#l4k = bbcast(l4k)
gln  =  bbcast(gln) 
ln  =  bbcast(ln) 


while g < 300:
   if pal<15:
    if is_l==False and rank==0:
           children1=comm.recv(source=node.index(nodes+'1'), tag=11)
           children2=comm.recv(source=node.index(nodes+'2'), tag=11)
           children3=comm.recv(source=node.index(nodes+'3'), tag=11)
           if pal==0:
              #ln = random.randint(0, gln)
              ln = ln-1
           if ln<0:
               ln=gln
           if ln==len(node[rank]):
            children=[]
           #cod,sols,fitnesses=ga2.kill_invlid_ind(cod,sols,fitnesses,l4k)
            for k in range(0,int(len(cod)*0.5)):
                 e=[cod[l4k[k][0]],cod[l4k[k][1]]]
                 e1,e2 = ga2.reproduction6(e)
                 children.append(e1)
                 children.append(e2)
            cod.extend(children)
           '''
           a= random.randint(0, 2)
           if a==0:
               cod=ga_f.update_pop2(cod,fitnesses)
           else:
               if a==1:
                 cod=ga_f.update_pop21(cod,fitnesses)
               else:
                      cod=ga_f.update_pop22(cod,fitnesses)
            '''     
           cod=ga_f.update_pop22(cod,fitnesses)
           sols=ga_f.decoding_pop(cod,children1,children2,children3)

           fitnesses=calc_fitns(sols,X)
           l4k=ga2.tour_list(cod,fitnesses)
           """
           zf=[]
           for k in range(0,int(len(cod)*0.5)):
              r =random.randint(0,len(lf)-1)
              zf.append(lf[r])
            """
           '''
           for i in range(len(sols)):
               sols[i]= tsp_2_opt(np.array(mt), sols[i])
               sols[i]=tsp_3_opt(np.array(mt), sols[i])
            '''
           #weakness_f=ga2.update_w(fitnesses)
           #cod,sols,fitnesses=ga2.kill_invlid_ind(cod,sols,fitnesses,l4k)
           fitnesses=calc_fitns(sols,X)
           
           if fbs>min(fitnesses):
                  fbs=min(fitnesses)
                  bs=sols[fitnesses.index(fbs)]
           print(fbs,g,ln,gln,len(bs))
    else:
         if is_l==False:
            if ln-1==len(node[rank]):
               children1=comm.recv(source=node.index(nodes+'1'), tag=11)
               children2=comm.recv(source=node.index(nodes+'2'), tag=11)
               children3=comm.recv(source=node.index(nodes+'3'), tag=11)
               #comm.send(sols, dest=node.index(nodes[:-1]), tag=11)
               children=[]
               lft=calc_fitns1(sols,X)
               l4k=ga2.tour_list(cod,lft)
               #zf=update_zf(zf)  
               for k in range(0,int(len(cod)*0.5)):
                 e=[cod[l4k[k][0]],cod[l4k[k][1]]]
                 #if zf.__contains__(nodes)==True:
                 e1,e2 = ga2.reproduction6(e)
                 #else:
                     #e1,e2=e[0],e[1]
                 children.append(e1)
                 children.append(e2)
               cod.extend(children)
               a= random.randint(0, 2)
               #a= random.random()
               if a==0:
                  cod=ga_f.update_pop2(cod,fitnesses)
                  #sols=ga_f.update_pop2(sols,fitnesses)
               else:
                  if a==1:
                      cod=ga_f.update_pop21(cod,fitnesses)
                      #sols=ga_f.update_pop21(sols,fitnesses)
                  else:
                          cod=ga_f.update_pop22(cod,fitnesses)
                          #sols=ga_f.update_pop22(sols,fitnesses)

               sols=ga_f.decoding_pop(cod,children1,children2,children3)
               comm.send(sols, dest=node.index(nodes[:-1]), tag=11)
            else:
                #if ln==len(node[rank]):
                          children1=comm.recv(source=node.index(nodes+'1'), tag=11)
                          children2=comm.recv(source=node.index(nodes+'2'), tag=11)
                          children3=comm.recv(source=node.index(nodes+'3'), tag=11)
                          
                          a= random.randint(0, 2)
                          #a= random.random()
                          if a==0:
                              cod=ga_f.update_pop2(cod,fitnesses)
                              #sols=ga_f.update_pop2(sols,fitnesses)
                          else:
                                  if a==1:
                                     cod=ga_f.update_pop21(cod,fitnesses)
                                     #sols=ga_f.update_pop21(sols,fitnesses)
                                  else:
                                       cod=ga_f.update_pop22(cod,fitnesses)
                                       #sols=ga_f.update_pop22(sols,fitnesses)
                                   
                          sols=ga_f.decoding_pop(cod,children1,children2,children3)
                          comm.send(sols, dest=node.index(nodes[:-1]), tag=11)
         else:
           if ln==len(node[rank]):
             children=[]
             lft=calc_fitns1(sols,X)
             l4k=ga2.tour_list(sols,lft)
             #print(l4k)
             for k in range(0,int(len(sols)*0.5)):
                 e=[sols[l4k[k][0]],sols[l4k[k][1]]] 
                 #if zf.__contains__(nodes)==True:
                 e1,e2 = ga2.reproduction5(e)
                 #else:
                     #e1,e2=e[0],e[1]
                 children.append(e1)
                 children.append(e2)
             sols.extend(children)
             a= random.randint(0, 2)
             #a= random.random()
             
             if a==0:
               sols=ga_f.update_pop2(sols,fitnesses)
             else:
                if a==1:
                    sols=ga_f.update_pop21(sols,fitnesses)
                else:
                       sols=ga_f.update_pop22(sols,fitnesses)
                   

             comm.send(sols, dest=node.index(nodes[:-1]), tag=11)
           else:
                
                 a= random.randint(0, 2)
                 #a= random.random()
                 if a==0:
                     sols=ga_f.update_pop2(sols,fitnesses)
                 else:
                   if a==1:
                      sols=ga_f.update_pop21(sols,fitnesses)
                   else:
                        sols=ga_f.update_pop22(sols,fitnesses)
                
                 comm.send(sols, dest=node.index(nodes[:-1]), tag=11)
 
          
   fitnesses = bbcast(fitnesses)
   #l4k = bbcast(l4k)
   ln= bbcast(ln)   
   g=g+1
   if pal==15: 
        pal=0
   else:
        pal=pal+1


if rank==0:
    print(bs,fbs)