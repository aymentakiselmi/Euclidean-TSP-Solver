import random
import itertools
import math
import kppv
from scipy.spatial import distance
import numpy as np
#from py2opt.routefinder import RouteFinder
import ga2
from aco import ACO, Graph

def generation_sols(n,m):
    #n = '1234'
    #n=''.join(list(map(str,n)))
    a=[]
    for i in itertools.permutations(n):
        a.append(list(i))
    '''
    sols=[]
    for i in a:
        sols.append(str2int(i))
   '''
    
    return a

def init_sol(d):
    i,j=0,0
    t=[]
    while i < d:
        j=random.randint(0,d-1)
        if t.__contains__(j)==False :
            t.append(j)
            i=i+1
    return t

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

def bnew(data,iu):
  exact=[]
  for i in range(0,len(data)):
        exact.append(data[iu[i]])
  data=exact.copy()
  return data

def gene(t):
      sol=generation_sols(t,len(t))     
      #s=random.randint(0,len(sol)-1)
      #sol
      return sol

def construct_sol_leaf(children):
        sol=[]
        points=children[0]
        for i in range(1,len(children)):
                points=points+children[i]
        sol=gene(points)
        sol=filter_sol(sol,children)
        return sol
def filter_sol(sol,children):
          new_sol=[]
          for i in range(len(sol)):
                  if invalide_sol(sol[i],children)==True:
                         new_sol.append(sol[i])
          return new_sol
def invalide_sol(sol,children):
         if  is_true(sol,children[0])==True and is_true(sol,children[1])==True  and is_true(sol,children[2])==True:
                return True
         else:
                return False

def is_true(sol,ind):
        if len(ind)==1:
            return True
        else:
            for i in range(len(sol)-1):
                 if (sol[i]==ind[0] and  sol[i+1]==ind[1]) or (sol[i]==ind[1] and  sol[i+1]==ind[0]):
                     return True
        return False
     
def init_pop_leaf(n,t,X): 
     new_leaf=[]
     cond=sous_m(t,len(t),X)
     r=random.randint(0,1)
     
     for i in range(n):
        
         if len(t)>2:
           pathh=kppv.init_sol2(len(cond),cond)
           #path=a2opt.tsp_2_opt(cond,ro)
           #path=init_sol(len(cond))   
           path=bnew(t,pathh)
         else:
             path=t
      
         
         #new_leaf.append(gene(t))
         new_leaf.append(path)
    
     random.shuffle(new_leaf)  
     return new_leaf

def init_pop_leaf1(n,t,X): 
     new_leaf=[]
     cond=sous_m(t,len(t),X)
     r=random.randint(0,1)
     if len(t)>2:
         aco=ACO(150, 20, 1.20, 0.75, 0.1, 1.0, 2)  
         graph = Graph(cond, len(cond))
         pathh, cost = aco.solve(graph)
     for i in range(n):
         
         if len(t)>2:
           #pathh=kppv.init_sol2(len(cond),cond)
           #path=a2opt.tsp_2_opt(cond,ro)
           #path=init_sol(len(cond))   
           path=bnew(t,pathh)
         else:
             path=t
      
         
         #new_leaf.append(gene(t))
         new_leaf.append(path)
   
     random.shuffle(new_leaf)  
     return new_leaf

def init_pop_leaf2(n,t,X): 
     new_leaf=[]
     sol=[]
     cond=sous_m(t,len(t),X)
     r=random.randint(0,1)
     for j in range(8):
       if len(t)>2:
         aco=ACO(150, 80, 0.98, 1.10, 0.1, 1.0, 2)  
         graph = Graph(cond, len(cond))
         pathh, cost = aco.solve(graph)
         path=bnew(t,pathh)
         sol.append(path)
       else:
           sol.append(t)
     for i in range(int(n/8)):
         #new_leaf.append(gene(t))
         new_leaf=new_leaf+sol
    
     random.shuffle(new_leaf)  
     return new_leaf

'''
def init_pop_not_leaf(n): 
     new_leaf=[['11','21','31'],
     ['11','21','32'],
     ['11','22','31'],
     ['11','22','32'],
     ['12','21','31'],
     ['12','22','31'],
     ['12','21','32'],
     ['12','22','32'],
     ['11','31','21'],
     ['11','31','22'],
     ['11','32','21'],
     ['11','32','22'],
     ['12','31','21'],
     ['12','31','22'],
     ['12','32','21'],
     ['12','32','22']
     ]
     for i in range(n):
         new_leaf=new_leaf+gene(t)
     random.shuffle(new_leaf)  
     return new_leaf
'''

def init_pop_not_leaf(n): 
     new_leaf1=[[1,2,0],[1,2,0],[1,2,0],[1,2,0],[1,2,0],[1,2,0],[1,2,0],[1,2,0],[0,2,1],[0,2,1],[0,2,1],[0,2,1],[0,2,1],[0,2,1],[0,2,1],[0,2,1]]
     '''
     new_leaf1=[]
     for i in range(n):
         new_leaf1.append(init_sol(3))
    '''
     new_leaf2=[[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,1,0],[1,0,1],[1,1,1],[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
     '''
     new_leaf2=[]
     for i in range(n):
         new_leaf2.append(list(np.random.choice([0, 1], size=(3,),p=[1./3, 2./3])))
     '''
     new_leaf3=list(zip(new_leaf1, new_leaf2))
     new_leaf=[]
     
     for i in new_leaf3:
           new_leaf.append(list(i))
     
     #new_leaf=[new_leaf1, new_leaf2]
     
     for i in range(n):
         new_leaf=new_leaf+new_leaf
     
     random.shuffle(new_leaf)  
     return new_leaf

def S_crossover(l, q):
    l=list(l)
    q=list(q)
    
  
    k = random.randint(0, len(l)-1)
    #print("Crossover point :", k)
  
# interchanging the genes
    for i in range(k, len(l)):
        l[i], q[i] = q[i], l[i]

    return l, q


def tournament_selection(pop,fitnesses,k):
 best = -1
 #fitnesses=[f_c1(chromosome,len(chromosome),nodelist) for chromosome in pop]
 new_pop=[]
 j=0
 while j < k:
  for i in range(0,3):
    ind = random.randint(0,len(pop)-1)
    if    best ==-1 or fitnesses[ind]  < fitnesses[best]:
          best = ind
  new_pop.append(pop[best])
  j=j+1
 return new_pop

def roulette_select(population, fitnesses, num):
    total_fitness = float(sum(fitnesses))
    rel_fitness = [1-(f/total_fitness) for f in fitnesses]
    # Generate probability intervals for each individual
    probs = [sum(rel_fitness[:i+1]) for i in range(len(rel_fitness))]
    # Draw new population
    new_population = []
    for n in range(num):
        r =random.uniform(0,1)
        for (i, individual) in enumerate(population):
            if r <= probs[i]:
                new_population.append(individual)
                break
    return new_population

def update_pop1(pop,child):
    #sorted(pop,key=fitness)
    new_pop=[]
    for i in range(0,int(len(pop)/2)):
        ind = random.randint(0,len(pop)-1)
        new_pop.append(pop[ind])
    new_pop=child+new_pop
    return new_pop

def update_pop(pop,fitness,child):
    #sorted(pop,key=fitness)
    pop = [x for _,x in sorted(zip(fitness,pop))]
    #print(pop)
    pop=pop[:int(len(pop)/2)]
    pop=pop+child
    return pop

def update_pop2(pop,fitness):
    #sorted(pop,key=fitness)
    pop = [x for _,x in sorted(zip(fitness,pop))]
    #print(pop)
    #x=random.randint(32,int(len(pop)/2))
    new_pop=[]
    for i in range(256):
           new_pop.append(pop[i])
    
    return new_pop
    

def update_pop2v(pop,fitness,children):
    #sorted(pop,key=fitness)
    pop = [x for _,x in sorted(zip(fitness,pop))]
    #print(pop)
    #x=random.randint(32,int(len(pop)/2))
    new_pop=[]
    size=2048-len(children)
    for i in range(size):
           new_pop.append(pop[i])
    new_pop.extend(children)
    return new_pop

def update_pop2s1(pop,fitness):
    #sorted(pop,key=fitness)
    pop = [x for _,x in sorted(zip(fitness,pop))]
    #print(pop)
    #x=random.randint(32,int(len(pop)/2))
    pop=pop[:128]
    
    return pop

def update_pop21(pop,fitness):
    #sorted(pop,key=fitness)
    pop = [x for _,x in sorted(zip(fitness,pop))]
    pop.reverse()
    #print(pop)
    #x=random.randint(32,int(len(pop)/2))
    new_pop=[]
    for i in range(256):
           new_pop.append(pop[i])
    
    return new_pop

def update_pop21v(pop,fitness,children):
    #sorted(pop,key=fitness)
    pop = [x for _,x in sorted(zip(fitness,pop))]
    pop.reverse()
    #print(pop)
    #x=random.randint(32,int(len(pop)/2))
    size=2048-len(children)
    new_pop=[]
    for i in range(size):
           new_pop.append(pop[i])
    
    new_pop.extend(children)
    return new_pop


def update_pop21s2(pop,fitness):
    #sorted(pop,key=fitness)
    pop = [x for _,x in sorted(zip(fitness,pop))]
    pop.reverse()
    #print(pop)
    #x=random.randint(32,int(len(pop)/2))
    pop=pop[:256]
    
    return pop

def update_pop22(pop,fitness):
    #sorted(pop,key=fitness)
    new_pop=[]
    pop = [x for _,x in sorted(zip(fitness,pop))]
    #print(pop)
    #x=random.randint(32,int(len(pop)/2))
    for i in range(0,256):
        #ind = random.randint(0,len(pop)-1)
        ind=ga2.tournament_selection1(pop, int(len(pop)* 0.3),fitness)
        
        new_pop.append(pop[ind])

    #pop=pop[:256]
    
    return new_pop

def update_pop22v(pop,fitness,children):
    #sorted(pop,key=fitness)
    new_pop=[]
    pop = [x for _,x in sorted(zip(fitness,pop))]
    #print(pop)
    #x=random.randint(32,int(len(pop)/2))
    size=2048-len(children)
    for i in range(0,size):
        #ind = random.randint(0,len(pop)-1)
        ind=ga2.tournament_selection1(pop, int(len(pop)* 0.3),fitness)
        
        new_pop.append(pop[ind])
    
    new_pop.extend(children)
    #pop=pop[:256]
    
    return new_pop

def update_pop22s3(pop,fitness):
    #sorted(pop,key=fitness)
    new_pop=[]
    pop = [x for _,x in sorted(zip(fitness,pop))]
    #print(pop)
    #x=random.randint(32,int(len(pop)/2))
    for i in range(0,128):
        #ind = random.randint(0,len(pop)-1)
        ind=ga2.tournament_selection1(pop, int(len(pop)* 0.3),fitness)
        
        new_pop.append(pop[ind])

    #pop=pop[:256]
    
    return new_pop


def M_inversion(p,pos1,pos2):
    token,i,k=0,pos1,pos2
    while i<=k:
        token=p[i]
        p[i]=p[k]
        p[k]=token
        i=i+1
        k=k-1
    return p

def M_deplacement(p,pos,pos1,pos2):
    j=pos
    token=0
    if len(p)-pos > pos2-pos1:
        for i in range(pos1,pos2+1):
                token=p[i]
                p[i]=p[j]
                p[j]=token
                j=j+1
    return p

def is_contain_cx(x,p1,p2):
       if  p2.__contains__(x) ==True:
         return p2[p1.index(x)]
       else:
          return -1

def list_cycle(x,p1,p2):
    z=[]
    z.append(x)
    while is_contain_cx(x,p1,p2)!=-1:
        x=is_contain_cx(x,p1,p2)
        if z.__contains__(x) ==True:
            break
        z.append(x)
    return z

def cx(p1,p2):
    l1,l2,e1,e2=[],[],[],[]
    e1=[0]*len(p1)
    e2=[0]*len(p1)
    i=0
    x=r=random.randint(0,len(p1)-1)
    #x=0
    x=p1[x]
    l1=list_cycle(x,p1,p2)
    #print(l1)
    #l2=list_cycle(x,p2,p1)
    while i < len(p1):
         if l1.__contains__(p1[i]) ==True:
                  e1[i]=p1[i]
                  e2[i]=p2[i]
         else:
                 e1[i]=p2[i]
                 e2[i]=p1[i]
         i=i+1
    return e1,e2

def cross(p1,p2):
    return p2,p1

def decoding_ind(cod,ind1,ind2,ind3):
      inds=[ind1,ind2,ind3]
      
      sol=[]
      for i in range(3):
          if i==0:
           if cod[1][i]==1:
               sol=sol+inds[cod[0][i]]
           else:
               sol=sol+list(reversed(inds[cod[0][i]]))
          else:
              if cod[1][i]==0:
                sol=sol+inds[cod[0][i]]
              else:
                sol=sol+list(reversed(inds[cod[0][i]]))
      return sol
def decoding_pop(cod,child1,child2,child3):
         sols=[]
         for i in range(len(cod)):
               sols.append(decoding_ind(cod[i],child1[i],child2[i],child3[i]))
         return sols

def inverse_solution(old_solution, i: int, j: int):
    numbers = [i, j]
    numbers.sort()
    i, j = numbers
    return old_solution[:i] + old_solution[i:j][::-1] + old_solution[j:]

def insert_solution(old_solution, i: int, j: int):
    new_solution = old_solution[:]
    new_solution.insert(j, new_solution[i])
    if j < i:
        i += 1 # because have inserted, the list is shifted by 1
    new_solution.pop(i)
    return new_solution

def mutation(indi):
        """
        Simple mutation.
        Arg:
            indi: individual to mutation.
        """
        point = np.random.randint(len(indi))
        indi[point] = 1 - indi[point]
        return indi

def partially_matched_crossover(ind1, ind2,size):
    #size = len(cities)
    p1, p2 = [0] * size, [0] * size

    # Initialize the position of each indices in the individuals
    for k in range(size):
        p1[ind1[k]] = k
        p2[ind2[k]] = k
    # Choose crossover points
    cxpoint1 = random.randint(0, size)
    cxpoint2 = random.randint(0, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

# Apply crossover between cx points
    for k in range(cxpoint1, cxpoint2):
    # Keep track of the selected values
        temp1 = ind1[k]
        temp2 = ind2[k]
    # Swap the matched value
        ind1[k], ind1[p1[temp2]] = temp2, temp1
        ind2[k], ind2[p2[temp1]] = temp1, temp2
    # Position bookkeeping
        p1[temp1], p1[temp2] = p1[temp2], p1[temp1]
        p2[temp1], p2[temp2] = p2[temp2], p2[temp1]

    return ind1, ind2

def fillNoneWithSwappedValue(arr1 ,arr2 ,final1 ,final2 ):
    for a in range(0,arr1.__len__()):
        if final1[a] == None:
            final1[a] = arr2[a]
        if final2[a] == None:
            final2[a] = arr1[a]
    return final1,final2

def indexOf(arr,x):
    for a in range(0,arr.__len__()):
        if arr[a] == x:
            return a
    return -1


def crossoverOperator( parent1, parent2 ):
    offspring1 = [None] * parent1.__len__()
    offspring2 = [None] * parent2.__len__()
    size1 = 1
    size2 = 1

    initalSelected = parent1[0]
    offspring1[0] = parent1[0]
    latestUpdated2 = parent2[0]
    check = 1

    while size1 < parent1.__len__() or size2 < parent2.__len__():
        if latestUpdated2 == initalSelected:
            index2 = indexOf(parent2,latestUpdated2)
            offspring2[index2] = parent2[index2]
            ans1,ans2 = fillNoneWithSwappedValue(parent1, parent2, offspring1, offspring2)
            offspring1 = ans1
            offspring2 = ans2
            size1 = parent1.__len__()
            size2 = parent2.__len__()
            check = 0
        else:
            index2 = indexOf(parent2,latestUpdated2)
            offspring2[index2] = parent2[index2]
            size2 += 1
            index1 = indexOf(parent1,parent2[index2])
            offspring1[index1] = parent1[index1]
            size1 += 1
            latestUpdated2 = parent2[index1]
    if check:
        index2 = indexOf(parent2, latestUpdated2)
        offspring2[index2] = parent2[index2]
    return offspring1,offspring2

def findUnusedIndexValues(parent,offspring):
    res = list()
    for a in parent:
        if indexOf(offspring,a) == -1:
            res.append(a)
    return res

def crossoverOperator2( parent1, parent2 ):
    #print('hellol shakoob')
    offspring1 = [None] * parent1.__len__()
    offspring2 = [None] * parent2.__len__()
    i1 = 0
    i2 = 0
    initalSelected = parent1[0]
    offspring1[i1] = parent2[0]
    i1 += 1
    # latestUpdated2 = parent2[0]
    check = 1

    while i1 < parent1.__len__() and i2 < parent2.__len__():
        index1 = indexOf(parent1,offspring1[i1-1])
        index1 = indexOf(parent1,parent2[index1])
        latestUpdated2 = parent2[index1]
        if latestUpdated2 == initalSelected:
            offspring2[i2] = latestUpdated2
            i2 += 1
            # print("cycle detected")
            check = 0
            res1 = findUnusedIndexValues(parent1,offspring1)
            res2 = findUnusedIndexValues(parent2,offspring2)
            # print(res1,res2)
            ans1,ans2 = crossoverOperator2(res1, res2)
            offspring1[i1:] = ans1
            offspring2[i2:] = ans2
            check = 0
            break
        else:
            offspring2[i2] = parent2[index1]
            i2 += 1
            index1 = indexOf(parent1,offspring2[i2-1])
            offspring1[i1] = parent2[index1]
            i1 += 1
    if check:
        index1 = indexOf(parent1, offspring1[i1 - 1])
        index1 = indexOf(parent1, parent2[index1])
        latestUpdated2 = parent2[index1]
        offspring2[i2] = latestUpdated2
        i2 += 1
    return offspring1,offspring2

def roulette_wheel_selection(fitness,population_size):
    s = 0
    partial_s = 0
    ind = 0
    for m in range(population_size):
        s = s + fitness[m]
    rand = random.uniform(0, s)
    for m in range(population_size):
        if partial_s < rand:
            partial_s = partial_s + fitness[m]
            ind = ind + 1
    if ind == population_size:  # prevent out of bounds list
        ind = population_size - 1
    return ind

'''
child1=init_pop_leaf(256,[14,15,17])
child2=init_pop_leaf(256,[20,19,18])
child3=init_pop_leaf(256,[13,11,12])
#cod=init_pop_not_leaf(4)
#print(decoding_pop(cod,child1,child2,child3))
print(child1[0],child2[0])
print(S_crossover(child1[0],child2[0]))
'''

'''
import imageio as iio
import cv2
# read an image
img = cv2.imread("g4g.jpeg")
img=cv2.resize(img,(224,224))
img=np.array(img)
img=img/255.0
print(img)
'''
