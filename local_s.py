import random

def two_opt(solution):
    current_solution=solution[:]
    # Select two random indices to swap
    index1 = random.randint(0, len(current_solution) - 2)
    index2 = random.randint(index1 + 1, len(current_solution) - 1)
    
    # Reverse the sub-list between the two indices
    current_solution[index1:index2] = current_solution[index1:index2][::-1]
    return current_solution


def three_opt(solution):  
    current_solution=solution[:]    
    # Select three random indices to swap
    index1 = random.randint(0, len(current_solution) - 3)
    index2 = random.randint(index1 + 1, len(current_solution) - 2)
    index3 = random.randint(index2 + 1, len(current_solution) - 1)
    
    # Rearrange the sub-list between the three indices
    current_solution[index1:index3] = current_solution[index1:index3][::-1] if index3 > index2 else current_solution[index3:index1:-1]
    return current_solution

          
def relocation(solution):
    current_solution=solution[:]
    # Select a random city from the current tour
    city_to_relocate = random.choice(current_solution)
    
    # Remove the city from the current tour
    current_solution.remove(city_to_relocate)

    # Insert the city at a new random position in the tour
    new_position = random.randint(0, len(current_solution))
    current_solution.insert(new_position, city_to_relocate)
    return current_solution

def exchange(current_solution):
    
    # Select two random cities from the current tour
    index1 = random.randint(0, len(current_solution) - 1)
    index2 = random.randint(0, len(current_solution) - 1)
    while index1 == index2:
        index2 = random.randint(0, len(current_solution) - 1)
    
    # Swap the positions of the two cities
    current_solution[index1], current_solution[index2] = current_solution[index2], current_solution[index1]

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

def swap_routes(route):
    """
    Moves a randomly selected subroute in the given route and inserts it at
    a randomly selected index.
    
    Args:
        route (list): The list representing the route, where each element is a node.
        
    Returns:
        list: The updated route with the subroute moved and inserted.
    """
    # Select a random subroute by selecting two random nodes (a and b) and
    # finding the indices of those nodes in the route
    a, b = random.sample(route, k=2)
    start_index = route.index(a)
    end_index = route.index(b)
    if start_index > end_index:
        start_index, end_index = end_index, start_index
    
    # Select a random index to insert the subroute at
    insert_index = random.randint(0, len(route) - (end_index - start_index) - 1)
    
    # Extract the subroute
    subroute = route[start_index:end_index+1]
    
    # Remove the subroute from the route
    route = route[:start_index] + route[end_index+1:]
    
    # Insert the subroute at the random index
    route = route[:insert_index] + subroute + route[insert_index:]
    
    return route
def scramble(current_solution):
    new_solution = current_solution.copy()
    
    i, j = random.sample(range(len(new_solution)), 2) # choose two indices for the segment
    while abs(i-j)>10:
         i, j = random.sample(range(len(new_solution)), 2)
    if i > j:
        i, j = j, i
    segment = new_solution[i:j+1]
    random.shuffle(segment) # shuffle the segment
    new_solution[i:j+1] = segment
    return new_solution

def reversal(current_solution):
    new_solution = current_solution.copy()
    i, j = random.sample(range(len(new_solution)), 2) # choose two indices for the segment
    if i > j:
        i, j = j, i
    new_solution[i:j+1] = reversed(new_solution[i:j+1])
    return new_solution

# Swap-Insert
def swap_insert(current_solution, mutation_rate):
    new_solution = current_solution.copy()
    i, j = random.sample(range(len(new_solution)), 2) # choose two indices to swap
    new_solution[i], new_solution[j] = new_solution[j], new_solution[i] # swap the values
    if random.random() < mutation_rate:
        k = random.randint(0, len(new_solution)) # choose an index to insert at
        value = random.choice([True, False]) # choose a value to insert
        new_solution.insert(k, value)
    return new_solution


def scatter_search():
    
    # Define the local search operators and their weights
    operators = ['two_opt', 'three_opt', 'inverse_solution','insert_solution','swap_solution','relocation']
    weights = [0.1,0.1,0.1,0.3,0.1,0.3]
    
    # Select a random operator and apply it to the current solution
    operator_to_apply = random.choices(operators, weights)[0]
    return operator_to_apply
          

'''
import numpy as np

def lam_cooling(T0, lambda_param, iteration):
    return T0 / (1 + lambda_param * np.log(iteration))

def hung_cooling(T0, lambda_param, iteration, total_iterations, beta):
    return T0 / (1 + lambda_param * (iteration/total_iterations)**beta)

# Example usage
T0 = 100
lambda_param = 0.8
iteration = 10
total_iterations = 100
beta = 0.5
'''
