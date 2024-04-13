import random
import ga_f

def generate_2_random_index(cities) -> int:
    return random.sample(range(len(cities)), 2)

def get_fittest_genome(genomes):
    genome_fitness = [genome[1] for genome in genomes]
    return genomes[genome_fitness.index(min(genome_fitness))][0]

def get_fittest_genome1(genomes):
    genome_fitness = [genome[1] for genome in genomes]
    return genome_fitness.index(min(genome_fitness))

def tournament_selection(population, k:int,fitnesses):
    p=list(zip(population,fitnesses))
    selected_genomes = random.sample(p, k)
    selected_parent = get_fittest_genome(selected_genomes)
    return selected_parent

def tournament_selection1(population, k:int,fitnesses):
    p=list(zip(population,fitnesses))
    selected_genomes = random.sample(p, k)
    selected_parent = get_fittest_genome1(selected_genomes)
    return selected_parent


def reproduction1(population,fitnesses):
    parents = [tournament_selection(population, int(len(population)* 0.2),fitnesses), random.choice(population)] 
    #parents = [tournament_selection(population, 20,fitnesses), random.choice(population)] 
    child1,child2=parents[0],parents[1]
    #if random.random() < 0.7:
    child1,child2 = ga_f.cx(child1,child2)
    if random.random() < 0.4:
        if len(child1)>2:
          i, j = generate_2_random_index(child1)
          r=random.randint(0,1)
          if r==0:
              child1=ga_f.inverse_solution(child1,i,j)
          else:
              child1=ga_f.insert_solution(child1,i,j)
          i, j = generate_2_random_index(child2)
          r=random.randint(0,1)
          if r==0:
               child2=ga_f.inverse_solution(child2,i,j)
          else:
              child2=ga_f.insert_solution(child2,i,j)

    return child1,child2

def reproduction3(parents):

    child1,child2=parents[0],parents[1]
    if random.random() < 0.75:
     if len(child1)>2:
         child1,child2 = ga_f.cx(child1,child2)
    if random.random() < 0.05:
        if len(child1)>2:
          i, j = generate_2_random_index(child1)
          r=random.randint(0,1)
          if r==0:
              child1=ga_f.inverse_solution(child1,i,j)
          else:
              child1=ga_f.insert_solution(child1,i,j)
          '''
          i, j = generate_2_random_index(child2)
          r=random.randint(0,1)
          if r==0:
               child2=ga_f.inverse_solution(child2,i,j)
          else:
              child2=ga_f.insert_solution(child2,i,j)
          '''
    if random.random() < 0.05:
        if len(child1)>2:
          i, j = generate_2_random_index(child2)
          r=random.randint(0,1)
          if r==0:
               child2=ga_f.inverse_solution(child2,i,j)
          else:
              child2=ga_f.insert_solution(child2,i,j)
    return child1,child2


def reproduction5(parents):

    child1,child2=parents[0],parents[1]
    if random.random() < 0.3:
     if len(child1)>2:
          i, j = generate_2_random_index(child1)
          r=random.randint(0,1)
          if r==0:
              child1=ga_f.inverse_solution(child1,i,j)
              child2=ga_f.inverse_solution(child2,i,j)

          else:
              child1=ga_f.insert_solution(child1,i,j)
              child2=ga_f.insert_solution(child2,i,j)

    return child1,child2


def reproduction2(population,fitnesses):
    parents = [tournament_selection(population, int(len(population)* 0.2),fitnesses), random.choice(population)] 
    #parents = [tournament_selection(population, 20,fitnesses), random.choice(population)] 
    child1,child2=parents[0],parents[1]
    #if random.random() < 0.7:
    child1[0],child2[0]=ga_f.cx(parents[0][0],parents[1][0])
    child1[1],child2[1]=ga_f.S_crossover(parents[0][1],parents[1][1])
    
    if random.random() < 0.4:
                      i, j = generate_2_random_index(child1[0])
                      r=random.randint(0,1)
                      if r==0:
                         child1[0]=ga_f.inverse_solution(child1[0],i,j)
                      else:
                          child1[0]=ga_f.insert_solution(child1[0],i,j)
                      i, j = generate_2_random_index(child1[0])
                      child1[1]=ga_f.mutation(child1[1])
                      r=random.randint(0,1)
                      if r==0:
                           child2[0]=ga_f.inverse_solution(child2[0],i,j)
                      else:
                           child2[0]=ga_f.insert_solution(child2[0],i,j)
                      i, j = generate_2_random_index(child1[0])
                      child2[1]=ga_f.mutation(child2[1])

    return child1,child2


def reproduction4(parents):
    
    child1,child2=parents[0],parents[1]
    #child1[1],child2[1]=ga_f.S_crossover(parents[0][1],parents[1][1])
    if random.random() < 0.75:
      child1[1],child2[1]=ga_f.cross(parents[0][1],parents[1][1])
      #child2[1]=ga_f.S_crossover(parents[1][1])
      #child1[1]=ga_f.mutation(child1[1])
      #child2[1]=ga_f.mutation(child2[1])
    if random.random() < 0.05:
                      
                      i, j = generate_2_random_index(child1[0])
                      r=random.randint(0,1)
                      if r==0:
                         child1[0]=ga_f.inverse_solution(child1[0],i,j)
                      else:
                          child1[0]=ga_f.insert_solution(child1[0],i,j)
                
                      i, j = generate_2_random_index(child1[0])
                      child1[1]=ga_f.mutation(child1[1]) 
                       
                      
        
                      '''
                      i, j = generate_2_random_index(child1[0])
                      child2[1]=ga_f.mutation(child2[1])
                      '''
                      #child1[0][0],child1[0][-1]=child1[0][-1],child1[0][0]
                      #child1[1]=ga_f.mutation(child1[1]) 
    if random.random() < 0.05:
                      
                      i, j = generate_2_random_index(child1[0])
                      r=random.randint(0,1)
                      if r==0:
                           child2[0]=ga_f.inverse_solution(child2[0],i,j)
                      else:
                           child2[0]=ga_f.insert_solution(child2[0],i,j)
                      child2[1]=ga_f.mutation(child2[1])

                      #child2[0][0],child2[0][-1]=child2[0][-1],child2[0][0]
                      #child2[1]=ga_f.mutation(child2[1])
    return child1,child2

def reproduction6(parents):
    
    child1,child2=parents[0],parents[1]
    #child1[1],child2[1]=ga_f.S_crossover(parents[0][1],parents[1][1])
    #child1[1],child2[1]=ga_f.S_crossover(parents[0][1],parents[1][1])
    r1=random.randint(0,1)
   
    child1[1]=ga_f.mutation(child1[1])                 
    child2[1]=ga_f.mutation(child2[1])
    
    i, j = generate_2_random_index(child1[0])
    r=random.randint(0,1)
    if r==0:
                         child1[0]=ga_f.inverse_solution(child1[0],i,j)
                         child2[0]=ga_f.inverse_solution(child2[0],i,j)
    else:
                          child1[0]=ga_f.insert_solution(child1[0],i,j)
                          child2[0]=ga_f.insert_solution(child2[0],i,j)

                     

    return child1,child2

def tour_list(population,fitnesses):
          l_p=[]
          for i in range(0,int(len(population)*0.5)):
                    #parents = [tournament_selection1(population, int(len(population)* 0.3),fitnesses), random.randint(0,len(population)-1)]
                    parents = [random.randint(0,len(population)-1), random.randint(0,len(population)-1)]
                    #parents = [tournament_selection1(population, int(len(population)* 0.3),fitnesses), tournament_selection1(population, int(len(population)* 0.3),fitnesses)]
                    #parents = [ga_f.roulette_wheel_selection(fitnesses,len(population)), ga_f.roulette_wheel_selection (fitnesses,len(population))]
                    '''
                    while True:
                         if parents[0] == parents[1]:
                                  #parents = [ga_f.roulette_wheel_selection(fitnesses,len(population)), ga_f.roulette_wheel_selection(fitnesses,len(population))]
                                  #parents = [random.randint(0,len(population)-1), random.randint(0,len(population)-1)]
                                  parents = [tournament_selection1(population, int(len(population)* 0.3),fitnesses), tournament_selection1(population, int(len(population)* 0.3),fitnesses)]
                         else:
                                  break
                    '''
                    l_p.append(parents)
          return l_p


def tour_list1(population,fitnesses):
          l_p=[]
          for i in range(0,int(len(population)*0.5)):
                    #parents = [tournament_selection1(population, int(len(population)* 0.2),fitnesses), random.randint(0,len(population)-1)]
                    parents = [tournament_selection1(population, int(len(population)* 0.3),fitnesses), tournament_selection1(population, int(len(population)* 0.3),fitnesses)]
                    #parents = [ga_f.roulette_wheel_selection(fitnesses,len(population)), ga_f.roulette_wheel_selection (fitnesses,len(population))]
                    '''
                    while True:
                         if parents[0] == parents[1]:
                                  parents = [ga_f.roulette_wheel_selection(fitnesses,len(population)), ga_f.roulwette_wheel_selection (fitnesses,len(population))]
                         else:
                                  break
                    '''
                    l_p.append(parents)
          return l_p
          
def list4kill_invlid_ind(fitnesses,w):
    indx=[]
    for ind in range(len(fitnesses)):
        if  fitnesses[ind] >= w:
              indx.append(ind)
    return indx
    
def kill_invlid_ind(cod,sols,fitnesses,l):
    kill=len(l)-1
    
    while kill >=0:
         fitnesses.remove(fitnesses[l[kill]])
         cod.remove(cod[l[kill]])
         sols.remove(sols[l[kill]])
         kill=kill-1
    return cod,sols,fitnesses

def kill_invlid_ind_l(sols,fitnesses,l):

    kill=len(l)-1 
    while kill >=0:
         fitnesses.remove(fitnesses[l[kill]])
         sols.remove(sols[l[kill]])
         kill=kill-1
    return sols,fitnesses

def update_w(fitnesses):
    '''
    mn=min(fitnesses)
    mx=max(fitnesses)
    mn=(mx+mn)/2
    '''
    fitnesses.sort()
    if len(fitnesses)>100:
        i=int(len(fitnesses)*0.5)
    else:
         i=int(len(fitnesses)*0.8)
    

    return  fitnesses[i]
