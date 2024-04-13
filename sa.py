import random
import math
import os
from scipy.spatial import distance
from model import *
from config import *
import local_s
import numpy as np
import lam

tin,tend,sop,fop=5.0, 0.0,[],math.inf
lambda_param = 0.8
beta=0.5
acceptance_rate=0.0
  
def calculate_acceptance_rate(accepted_solutions, proposed_solutions):
    acceptance_rate = accepted_solutions / proposed_solutions
    return acceptance_rate

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

def f_c11(traget,n,nodelist):
    s=0.0
    i=0
    if traget!=[]:
     if len(traget)==1:
        s=traget[0]
     else:
      for i in range (0,n-1):
        s=s+ round(distance.euclidean(nodelist[traget[i]],nodelist[traget[i+1]]))
      s=s+ round(distance.euclidean(nodelist[traget[-1]],nodelist[traget[0]]))
      #s=s+ nodelist[traget[-1]][traget[0]]
    return s


def metropolis(s,new_s,t,a,i):
    fs=f_c11(s,len(s),X)
    fnew_s=f_c11(new_s,len(new_s),X)
    df=fnew_s-fs
    global fop
    global sop
    if df<0:
        print(fnew_s,i,t)
        if fnew_s < fop:
            sop=new_s
            fop= fnew_s
        a=True
        return new_s,a
    else:
      if random.uniform(0,1) <(math.exp(-df/(t))):
                a=True
                return new_s,a
      else:
                return s,False

def cooling_sch(t,i):
    t=tin + t/i *(tend-tin)
    return t

def cooling_sch1(t):
    t=t*0.9999
    return t
def inverse_solution(old_solution, i: int, j: int) :
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

def swap_solution(old_solution, i: int, j: int):
    new_solution = old_solution[:]
    temp = new_solution[i]
    new_solution[i] = new_solution[j]
    new_solution[j] = temp
    return new_solution
'''
def generate_2_random_index(cities) -> int:
    i=random.randint(1,len(cities)-2)
    j= random.randint(i-1,i+1)
    return i,j
'''

def generate_2_random_index(cities: List[City]) -> int:
    return random.sample(range(len(cities)), 2)

def shift(solution, amount):
    # Shift the solution by the given amount
    shifted_solution = solution[amount:] + solution[:amount]
    return shifted_solution

def create_new_solution(cities, old_solution, i_test: int = -1, j_test: int = -1):
    # helper for unit test, so number is not random
    i, j = i_test, j_test 

    if i == -1 or j == -1:
        i, j = generate_2_random_index(cities)
    #print(i,j)
    inverse_opt = inverse_solution(old_solution, i, j)
    insert_opt = insert_solution(old_solution, i, j)
    swap_opt = swap_solution(old_solution, i, j)
    two_opt=local_s.two_opt(old_solution)
    three_opt=local_s.three_opt(old_solution)
    scrm=local_s.scramble (old_solution)
    
    #two_opt =kopt.run_kopt(old_solution,cond)
    r_opt=local_s.relocation(old_solution)
    #r_opt=shift(old_solution,i)
    evaluation = [f_c11(inverse_opt,len(inverse_opt),cities), f_c11( insert_opt,len(insert_opt),cities), f_c11(swap_opt,len(swap_opt),cities), f_c11(two_opt,len(two_opt),cities), f_c11(three_opt,len(three_opt),cities), f_c11(r_opt,len(r_opt),cities),f_c11(scrm,len(scrm),cities)]
    #evaluation = [f_c11(inverse_opt,len(inverse_opt),cities), f_c11( insert_opt,len(insert_opt),cities), f_c11(swap_opt,len(swap_opt),cities),f_c11(scrm,len(swap_opt),cities)]
    r=random.uniform(0,1)
    if r<0.95:
        index = evaluation.index(min(evaluation))
    else:
        index=random.randint(0,len(evaluation)-1)

    if index == 0:
        return inverse_opt
    elif index == 1:
        return insert_opt
    else:
        if index == 2: 
            return swap_opt
        else:
            if index==3:
                return two_opt
            
            else:
                    if index==4:
                        return three_opt
                    else:
                        if index==5:
                             return r_opt
                        else:
                            return scrm

def create_new_solution1(cities, old_solution, i_test: int = -1, j_test: int = -1):
    # helper for unit test, so number is not random
    i, j = i_test, j_test 

    if i == -1 or j == -1:
        i, j = generate_2_random_index(cities)
    #print(i,j)
    #o=local_s.scatter_search()
    o=local_s.scatter_search()
    #o='two_opt'
    if o=='two_opt':
         s=local_s.two_opt(old_solution)
    if o=='three_opt':
         s=local_s.three_opt(old_solution)
    if o=='inverse_solution':
         s=inverse_solution(old_solution, i, j) 
    if o=='insert_solution':
         s=insert_solution(old_solution, i, j) 
    if o=='swap_solution':
         s=swap_solution(old_solution, i, j)    
    
    if o=='relocation':
         s=local_s.relocation(old_solution) 

    return s


def hung_cooling(T0, lambda_param, iteration, total_iterations, beta):
    return T0 / (1 + lambda_param * (iteration/total_iterations)**beta)


def hung_harming(T0, lambda_param, iteration, total_iterations, beta):
    return T0 *(1 + lambda_param * (iteration/total_iterations)**beta)

def lam_cooling(T0, lambda_param, iteration):
    return T0 / (1 + lambda_param * np.log(iteration))



def calculate_acceptance_rate(accepted_solutions, proposed_solutions):
    acceptance_rate = accepted_solutions / proposed_solutions
    return acceptance_rate

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
            dist=distance.euclidean(nodelist[traget[j]],nodelist[traget[i]])
            dist=math.sqrt((nodelist[traget[j]][0] - nodelist[traget[i]][0]) ** 2 + (nodelist[traget[j]][1] - nodelist[traget[i]][1]) ** 2)
            dist=round(dist)
            s.append(dist)
       ss.append(s)
       s=[]
    return ss

def anealing(): 
    
    sol=[3, 28, 44, 26, 51, 12, 53, 56, 14, 4, 36, 19, 69, 59, 70, 35, 68, 60, 27, 20, 46, 47, 73, 29, 67, 5, 50, 15, 62, 32, 1, 61, 72, 0, 21, 63, 41, 40, 42, 55, 22, 48, 23, 17, 43, 2, 16, 75, 74, 25, 66, 33, 45, 6, 34, 7, 18, 52, 13, 58, 65, 64, 37, 10, 71, 57, 9, 30, 54, 24, 49, 31, 8, 38, 39, 11]
    sol=[11, 39, 38, 8, 31, 49, 24, 54, 30, 9, 57, 71, 37, 64, 10, 65, 58, 13, 52, 18, 7, 34, 6, 45, 33, 66, 74, 75, 25, 16, 4, 36, 19, 69, 59, 70, 35, 68, 14, 56, 12, 53, 51, 26, 44, 28, 3, 29, 73, 47, 46, 20, 60, 27, 32, 62, 15, 50, 5, 67, 1, 61, 72, 0, 21, 63, 41, 40, 42, 55, 22, 48, 23, 17, 43, 2]
    sol= [0, 84, 144, 190, 26, 197, 122, 14, 12, 78, 159, 161, 63, 19, 134, 41, 54, 66, 176, 64, 186, 150, 79, 160, 124, 180, 1, 34, 22, 172, 167, 184, 61, 82, 71, 129, 128, 133, 21, 74, 53, 5, 108, 106, 156, 30, 46, 119, 185, 126, 111, 154, 182, 7, 16, 24, 33, 89, 142, 145, 102, 113, 97, 57, 140, 170, 199, 87, 147, 27, 38, 37, 70, 55, 151, 177, 195, 4, 104, 42, 136, 132, 175, 112, 194, 181, 85, 138, 49, 94, 93, 90, 149, 143, 69, 75, 101, 153, 20, 139, 163, 168, 67, 29, 76, 157, 192, 127, 59, 166, 40, 88, 58, 72, 2, 68, 141, 188, 130, 179, 155, 99, 32, 44, 80, 96, 118, 91, 9, 174, 98, 18, 189, 178, 117, 15, 62, 50, 193, 121, 169, 115, 187, 43, 152, 65, 47, 83, 10, 51, 86, 125, 95, 165, 164, 103, 196, 35, 56, 73, 107, 13, 191, 100, 3, 162, 92, 105, 148, 48, 17, 109, 28, 183, 36, 123, 137, 8, 77, 81, 6, 198, 25, 60, 135, 31, 23, 158, 173, 120, 171, 45, 11, 146, 39, 131, 110, 116, 114, 52]
    #sol=[i for i in range(len(X))]
    t=tin
    #adaptive_lam = lam.AdaptiveLamSchedule(initial_temp=1.0, acceptance_ratio_target=0.5, acceptance_ratio_window=10, max_iterations=10000)
    #t=adaptive_lam.get_temperature()
    global sop
    sop=sol
    global fop 
    fop=f_c11( sop,len(sop),X)
    fp=fop
    i=0     
    iter=25000
    a=0
    accepted,accepted_rate=1,0.0
    l_acc=0.4
    h_acc=0.6
    step_size=1.0
    step=0
    while i < iter :
         
         new_s=create_new_solution(X,sol)   
         sol,a=metropolis(sol,new_s,t,a,i)
         #t=cooling_sch1(t)
         if a==True: 
             accepted=accepted+1
         accepted_rate=accepted/iter
         #t=hung_cooling(tin,lambda_param,i,iter,beta)
         t=cooling_sch1(t)
         '''
         if accepted_rate<= l_acc:
              t=hung_harming(tin,lambda_param,i,iter,beta)
         if accepted_rate>= h_acc:
               t=hung_cooling(tin,lambda_param,i,iter,beta)
         step_size = math.sqrt(step_size * step_size * acceptance_rate / 0.5)
         '''
         ''''
         if fp <= f_c11( sop,len(sop),X):
               step=step+1
               if step==500:
                   sol=local_s.scramble(sol)
                   print("atmen")
                   step=0
         else:
                  step=0
                  fp=f_c11( sol,len(sop),X)
           '''         
                
         #adaptive_lam.update_counts(a)
         #t=adaptive_lam.get_temperature()
         #t=adaptive_hung_cooling_warming_up(i,iter,1.0,0,calculate_acceptance_rate(a,10))
         #t=lam_cooling(tin,lambda_param,i)
         #print(t)
         i=i+1
         
    print(sop,f_c11( sop,len(sop),X),i)
    print(sol)
X=read_data(DATA_FILE)
cond=sous_m([i for i in range(len(X))],len(X),X)
anealing()

