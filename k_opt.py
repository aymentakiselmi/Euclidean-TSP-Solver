from itertools import cycle, islice, dropwhile
from  opt_case import OptCase
from config import *
import os
from scipy.spatial import distance
import numpy as np
import random
import math
from aco import ACO, Graph
#import concorde.tsp as tsp

def _swap_2opt(route, i, k):
    """ Swapping the route """
    new_route = route[0:i]
    new_route.extend(reversed(route[i:k + 1]))
    new_route.extend(route[k + 1:])
    return new_route

def route_cost(graph, path):
    cost = 0.0
    for index in range(len(path) - 1):
        cost = cost + graph[path[index]][path[index + 1]]
    # add last edge to form a cycle.
    cost = cost + graph[path[-1], path[0]]
    return cost

def tsp_2_opt(graph, route):
    """
    Approximate the optimal path of travelling salesman according to 2-opt algorithm
    Args:
        graph: 2d numpy array as graph
        route: list of nodes
    Returns:
        optimal path according to 2-opt algorithm
    Examples:
        >>> import numpy as np
        >>> graph = np.array([[  0, 300, 250, 190, 230],
        >>>                   [300,   0, 230, 330, 150],
        >>>                   [250, 230,   0, 240, 120],
        >>>                   [190, 330, 240,   0, 220],
        >>>                   [230, 150, 120, 220,   0]])
        >>> tsp_2_opt(graph)
    """
    improved = True
    im=0
    best_found_route = route
    best_found_route_cost = route_cost(graph, best_found_route)
    while im<1:
        improved = False
        for i in range(1, len(best_found_route) - 1):
            for k in range(i + 1, len(best_found_route) - 1):
                new_route = _swap_2opt(best_found_route, i, k)
                new_route_cost = route_cost(graph, new_route)
                if new_route_cost < best_found_route_cost:
                    best_found_route_cost = new_route_cost
                    best_found_route = new_route
                    improved = True
                    #break
        im=im+1
        if improved:
               # break
               pass
    return best_found_route

def tsp_3_opt(graph, route):
    """
    Approximate the optimal path of travelling salesman according to 3-opt algorithm
    Args:
        graph:  2d numpy array as graph
        route: route as ordered list of visited nodes. if no route is given, christofides algorithm is used to create one.
    Returns:
        optimal path according to 3-opt algorithm
    Examples:
        >>> import numpy as np
        >>> graph = np.array([[  0, 300, 250, 190, 230],
        >>>                   [300,   0, 230, 330, 150],
        >>>                   [250, 230,   0, 240, 120],
        >>>                   [190, 330, 240,   0, 220],
        >>>                   [230, 150, 120, 220,   0]])
        >>> tsp_3_opt(graph)
    """

    moves_cost = {OptCase.opt_case_1: 0, OptCase.opt_case_2: 0,
                  OptCase.opt_case_3: 0, OptCase.opt_case_4: 0, OptCase.opt_case_5: 0,
                  OptCase.opt_case_6: 0, OptCase.opt_case_7: 0, OptCase.opt_case_8: 0}
    improved = True
    best_found_route = route
    im=0
    while im<1:
        improved = False
        for (i, j, k) in possible_segments(len(graph)):
            # we check all the possible moves and save the result into the dict
            for opt_case in OptCase:
                moves_cost[opt_case] = get_solution_cost_change(graph, best_found_route, opt_case, i, j, k)
            # we need the minimum value of substraction of old route - new route
            best_return = max(moves_cost, key=moves_cost.get)
            if moves_cost[best_return] > 0:
                best_found_route = reverse_segments(best_found_route, best_return, i, j, k)
                improved = True
                #break
        im=im+1
    # just to start with the same node -> we will need to cycle the results.
    cycled = cycle(best_found_route)
    skipped = dropwhile(lambda x: x != 0, cycled)
    sliced = islice(skipped, None, len(best_found_route))
    best_found_route = list(sliced)
    return best_found_route


def possible_segments(N):
    """ Generate the combination of segments """
    segments = ((i, j, k) for i in range(N) for j in range(i + 2, N-1) for k in range(j + 2, N - 1 + (i > 0)))
    return segments


def get_solution_cost_change(graph, route, case, i, j, k):
    """ Compare current solution with 7 possible 3-opt moves"""
    A, B, C, D, E, F = route[i - 1], route[i], route[j - 1], route[j], route[k - 1], route[k % len(route)]
    if case == OptCase.opt_case_1:
        # first case is the current solution ABC
        return 0
    elif case == OptCase.opt_case_2:
        # second case is the case A'BC
        return graph[A, B] + graph[E, F] - (graph[B, F] + graph[A, E])
    elif case == OptCase.opt_case_3:
        # ABC'
        return graph[C, D] + graph[E, F] - (graph[D, F] + graph[C, E])
    elif case == OptCase.opt_case_4:
        # A'BC'
        return graph[A, B] + graph[C, D] + graph[E, F] - (graph[A, D] + graph[B, F] + graph[E, C])
    elif case == OptCase.opt_case_5:
        # A'B'C
        return graph[A, B] + graph[C, D] + graph[E, F] - (graph[C, F] + graph[B, D] + graph[E, A])
    elif case == OptCase.opt_case_6:
        # AB'C
        return graph[B, A] + graph[D, C] - (graph[C, A] + graph[B, D])
    elif case == OptCase.opt_case_7:
        # AB'C'
        return graph[A, B] + graph[C, D] + graph[E, F] - (graph[B, E] + graph[D, F] + graph[C, A])
    elif case == OptCase.opt_case_8:
        # A'B'C
        return graph[A, B] + graph[C, D] + graph[E, F] - (graph[A, D] + graph[C, F] + graph[B, E])

def reverse_segments(route, case, i, j, k):
    """
    Create a new tour from the existing tour
    Args:
        route: existing tour
        case: which case of opt swaps should be used
        i:
        j:
        k:
    Returns:
        new route
    """
    if (i - 1) < (k % len(route)):
        first_segment = route[k% len(route):] + route[:i]
    else:
        first_segment = route[k % len(route):i]
    second_segment = route[i:j]
    third_segment = route[j:k]

    if case == OptCase.opt_case_1:
        # first case is the current solution ABC
        pass
    elif case == OptCase.opt_case_2:
        # A'BC
        solution = list(reversed(first_segment)) + second_segment + third_segment
    elif case == OptCase.opt_case_3:
        # ABC'
        solution = first_segment + second_segment + list(reversed(third_segment))
    elif case == OptCase.opt_case_4:
        # A'BC'
        solution = list(reversed(first_segment)) + second_segment + list(reversed(third_segment))
    elif case == OptCase.opt_case_5:
        # A'B'C
        solution = list(reversed(first_segment)) + list(reversed(second_segment)) + third_segment
    elif case == OptCase.opt_case_6:
        # AB'C
        solution = first_segment + list(reversed(second_segment)) + third_segment
    elif case == OptCase.opt_case_7:
        # AB'C'
        solution = first_segment + list(reversed(second_segment)) + list(reversed(third_segment))
    elif case == OptCase.opt_case_8:
        # A'B'C
        solution = list(reversed(first_segment)) + list(reversed(second_segment)) + list(reversed(third_segment))
    return solution

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
def init_sol(d):
    i,j=0,0
    t=[]
    while i < d:
        j=random.randint(0,d-1)
        if t.__contains__(j)==False :
            t.append(j)
            i=i+1
    return t



#path=[0, 21, 30, 17, 2, 16, 20, 41, 6, 1, 29, 22, 19, 49, 28, 15, 45, 43, 33, 34, 35, 38, 39, 37, 36, 47, 23, 4, 14, 5, 3, 24, 11, 27, 26, 25, 46, 13, 12, 51, 10, 50, 32, 42, 9, 8, 7, 40, 18, 44, 31, 48]
#path=[51, 12, 13, 46, 25, 26, 27, 50, 11, 10, 32, 42, 9, 8, 7, 40, 18, 44, 0, 31, 48, 35, 34, 33, 38, 36, 39, 37, 14, 4, 23, 47, 5, 3, 24, 45, 43, 15, 28, 49, 19, 22, 29, 20, 21, 30, 17, 2, 16, 41, 1, 6]
#path=[0, 48, 31, 44, 18, 40, 7, 8, 9, 42, 32, 50, 11, 10, 51, 13, 12, 46, 25, 26, 27, 24, 3, 5, 14, 4, 23, 47, 45, 36, 37, 39, 38, 35, 34, 33, 43, 15, 28, 49, 19, 22, 29, 1, 6, 41, 20, 16, 2, 17, 30, 21]
#path= [0, 21, 30, 17, 2, 16, 20, 41, 6, 1, 29, 22, 19, 49, 28, 15, 45, 43, 33, 34, 35, 38, 39, 36, 37, 47, 23, 4, 14, 5, 3, 24, 11, 27, 26, 25, 46, 12, 13, 51, 10, 50, 32, 42, 9, 8, 7, 40, 18, 44, 31, 48]
path=[47, 29, 73, 46, 20, 27, 60, 68, 35, 70, 69, 59, 19, 36, 4, 14, 56, 53, 12, 51, 26, 28, 44, 3, 52, 13, 58, 65, 10, 64, 37, 6, 34, 7, 18, 45, 33, 66, 25, 74, 75, 16, 57, 71, 9, 30, 54, 24, 49, 31, 8, 38, 11, 39, 2, 43, 17, 23, 48, 22, 55, 42, 40, 41, 63, 21, 0, 72, 61, 1, 32, 62, 15, 50, 5, 67]
path= [1,33,63,16,3,44,32,9,39,72,58,10,31,55,25,50,18,24,49,23,56,41,43,42,64,22,61,21,47,36,69,71,60,70,20,37,5,15,57,13,54,19,14,59,66,65,38,11,53,7,35,8,46,34,52,27,45,29,48,30,4,75,76,67,26,12,40,17,51,6,68,2,74,28,62,73]
path=[47, 169, 121, 115, 187, 43, 62, 193, 50, 15, 117, 178, 152, 65, 118, 18, 91, 174, 9, 98, 13, 191, 107, 59, 100, 3, 162, 92, 105, 148, 189, 48, 17, 109, 28, 183, 36, 123, 137, 8, 77, 81, 6, 198, 25, 60, 135, 31, 23, 158, 173, 120, 171, 45, 11, 146, 39, 131, 110, 116, 114, 52, 0, 84, 144, 190, 26, 197, 122, 14, 12, 78, 159, 176, 64, 79, 76, 157, 192, 127, 166, 29, 67, 168, 34, 1, 180, 124, 160, 150, 186, 53, 5, 108, 106, 156, 119, 46, 30, 66, 161, 63, 19, 54, 41, 134, 185, 126, 111, 154, 182, 74, 133, 21, 7, 16, 24, 142, 89, 33, 102, 145, 128, 113, 97, 57, 140, 170, 199, 87, 147, 27, 38, 37, 70, 129, 71, 82, 61, 184, 167, 49, 138, 85, 195, 55, 151, 177, 4, 104, 42, 136, 132, 175, 112, 194, 181, 93, 94, 90, 149, 172, 22, 143, 69, 75, 101, 153, 20, 139, 163, 88, 40, 58, 2, 72, 188, 68, 141, 130, 179, 73, 56, 35, 99, 155, 32, 44, 196, 80, 96, 103, 164, 165, 95, 125, 86, 51, 10, 83]
#path=init_sol(52)
path= [87, 147, 27, 195, 55, 151, 177, 4, 104, 42, 136, 132, 175, 112, 194, 181, 93, 38, 37, 70, 129, 71, 82, 61, 184, 167, 49, 138, 85,94, 90, 149, 172, 22, 143, 69, 75, 101, 163, 139, 20, 153, 88, 40, 58, 2, 72, 188, 68, 141, 130, 179, 155, 32, 99, 73, 56, 35, 13, 191, 107, 59, 100, 3, 162, 92, 105, 148, 189, 48, 17, 109, 28, 183, 36, 178, 152, 65, 118, 18, 98, 91, 9, 174, 196, 44, 80, 96, 103, 164, 165, 95, 125, 86, 51, 10, 83, 47, 169, 121, 115, 187, 43, 62, 193, 50, 15, 117, 123, 137, 8, 77, 81, 6, 198, 25, 60, 135, 31, 23, 158, 173, 120, 171, 45, 11, 146, 39, 131, 110, 116, 114, 52, 0, 84, 144, 190, 26, 197, 122, 14, 12, 78, 159, 176, 64, 79, 76, 157, 192, 127, 166, 29, 67, 168, 34, 1, 180, 124, 160, 150, 186, 53, 5, 108, 106, 156, 119, 46, 30, 66, 161, 63, 19, 54, 41, 134, 185, 126, 111, 154, 182, 74, 133, 21, 7, 16, 24, 142, 145, 128, 10289, 33, 102, 145, 128, 113, 97, 57, 140, 170, 199,176,65,79,76,157,192,127,166,29,67,168,34,1,180,124,160,150,186,53,5,108,106,156]
#path=  [201, 213, 284, 215, 335, 128, 119, 246, 221, 114, 303, 38, 294, 33, 160, 44, 255, 168, 3, 200, 8, 0, 202, 274, 272, 55, 136, 299, 338, 156, 91, 357, 366, 21, 256, 65, 214, 226, 56, 178, 95, 102, 107, 353, 359, 298, 384, 138, 1, 121, 208, 264, 381, 72, 211, 4, 254, 265, 352, 13, 140, 259, 279, 280, 225, 227, 79, 336, 376, 387, 222, 249, 268, 27, 377, 36, 74, 290, 369, 142, 333, 309, 302, 32, 323, 176, 143, 197, 77, 207, 276, 321, 350, 373, 291, 324, 301, 85, 371, 282, 182, 45, 137, 149, 293, 170, 104, 354, 247, 317, 169, 217, 26, 125, 130, 375, 42, 11, 320, 62, 296, 175, 220, 261, 124, 30, 241, 9, 188, 145, 372, 392, 15, 360, 206, 2, 92, 134, 329, 386, 194, 313, 115, 389, 271, 331, 131, 10, 99, 362, 244, 223, 58, 31, 361, 253, 61, 328, 344, 59, 147, 292, 109, 343, 326, 157, 306, 129, 310, 250, 275, 396, 319, 67, 395, 100, 349, 112, 141, 22, 190, 123, 47, 252, 46, 122, 205, 34, 270, 183, 35, 278, 153, 71, 322, 118, 378, 345, 229, 212, 341, 48, 57, 135, 5, 150, 287, 97, 239, 23, 305, 88, 267, 316, 172, 12, 18, 383, 297, 41, 390, 117, 195, 106, 159, 63, 6, 340, 315, 68, 75, 29, 20, 14, 300, 25, 327, 166, 163, 185, 16, 66, 204, 356, 210, 283, 76, 198, 233, 394, 53, 399, 288, 347, 277, 86, 311, 186, 355, 388, 242, 339, 40, 342, 70, 152, 126, 184, 251, 87, 243, 237, 50, 164, 89, 231, 382, 101, 260, 224, 286, 314, 113, 391, 365, 64, 83, 380, 173, 199, 330, 24, 162, 28, 196, 368, 94, 52, 397, 189, 307, 158, 84, 358, 155, 191, 203, 209, 179, 295, 151, 393, 49, 82, 318, 90, 148, 7, 334, 192, 337, 398, 364, 17, 304, 232, 181, 216, 139, 187, 266, 171, 133, 363, 69, 80, 240, 93, 234, 60, 332, 218, 73, 96, 379, 54, 167, 39, 245, 108, 263, 111, 110, 103, 98, 385, 269, 81, 105, 273, 19, 367, 165, 370, 146, 289, 351, 132, 248, 238, 37, 127, 43, 180, 144, 174, 228, 258, 161, 312, 348, 78, 374, 120, 236, 257, 285, 346, 177, 308, 193, 325, 230, 262, 235, 51, 116, 281, 154, 219]
path=[0, 202, 274, 272, 55, 136, 299, 338, 156, 91, 357, 366, 21, 256, 65, 214, 226, 56, 336, 225, 102, 178, 95, 107, 353, 359, 298, 384, 138, 1, 121, 208, 264, 381, 72, 211, 4, 254, 265, 352, 51, 13, 140, 259, 279, 280, 227, 79, 376, 387, 222, 249, 268, 27, 377, 105, 325, 81, 36, 74, 269, 273, 19, 367, 165, 370, 146, 289, 351, 132, 248, 238, 37, 127, 43, 180, 144, 174, 228, 258, 161, 312, 348, 78, 374, 120, 257, 236, 89, 231, 382, 101, 260, 224, 286, 314, 113, 391, 365, 64, 83, 380, 173, 199, 330, 24, 162, 28, 196, 368, 94, 52, 397, 189, 307, 158, 84, 358, 155, 191, 203, 209, 179, 295, 151, 393, 49, 82, 318, 90, 148, 7, 334, 192, 337, 398, 364, 17, 304, 232, 181, 216, 139, 187, 266, 171, 133, 363, 69, 80, 240, 93, 234, 60, 332, 218, 73, 96, 379, 263, 54, 167, 108, 245, 39, 111, 103, 110, 385, 98, 142, 333, 309, 302, 247, 354, 104, 193, 32, 323, 176, 143, 197, 77, 207, 276, 321, 350, 373, 291, 324, 301, 85, 371, 282, 182, 45, 137, 149, 293, 170, 317, 369, 290, 219, 230, 169, 217, 26, 125, 130, 375, 42, 11, 320, 62, 296, 175, 220, 261, 124, 30, 241, 9, 188, 145, 372, 392, 15, 360, 206, 2, 92, 134, 329, 386, 194, 313, 115, 389, 271, 331, 131, 10, 99, 362, 244, 223, 58, 31, 361, 253, 61, 328, 344, 59, 147, 292, 109, 343, 326, 157, 306, 129, 310, 250, 275, 396, 319, 67, 395, 100, 349, 112, 141, 22, 190, 123, 47, 252, 46, 122, 205, 34, 270, 183, 35, 278, 153, 71, 322, 118, 378, 345, 229, 212, 341, 48, 57, 135, 5, 150, 287, 97, 239, 23, 305, 88, 267, 316, 172, 12, 18, 383, 297, 41, 390, 117, 195, 106, 159, 63, 6, 340, 315, 68, 75, 29, 20, 14, 300, 25, 327, 166, 163, 281, 116, 185, 16, 66, 204, 356, 154, 235, 262, 210, 283, 76, 198, 233, 394, 53, 399, 288, 347, 277, 86, 311, 186, 355, 388, 242, 339, 40, 342, 70, 152, 126, 184, 251, 87, 243, 50, 164, 237, 308, 177, 346, 285, 213, 201, 284, 215, 335, 128, 119, 246, 38, 294, 33, 160, 44, 255, 168, 3, 303, 221, 114, 8, 200]
path= [ 76, 78, 95, 127, 138, 148, 154, 160, 162, 169, 172, 170, 174, 180, 189, 200, 175, 173, 179, 193, 185, 188, 184, 178, 165, 166, 161, 159, 155, 158, 153, 151, 144, 131, 125, 121, 120, 124, 128, 135, 145, 146, 147, 149, 143, 141, 139, 137, 129, 133, 132, 130, 134, 136, 126, 122, 119, 113, 114, 107, 107, 111, 111, 112, 118, 116, 106, 105, 98, 100, 97, 96, 0, 1, 3, 4, 5, 8, 9, 10, 6, 7, 11, 18, 17, 16, 15, 14, 13, 12, 52, 56, 55, 53, 54, 64, 57, 58, 59, 63,115, 86, 85, 84, 83, 62, 61, 60, 19, 65, 66, 68, 67, 69, 71, 70, 72, 73, 77, 80, 79, 82, 81, 75, 74,89, 87, 88, 90, 93, 91, 92, 94, 94, 99, 99, 104, 104, 101, 116, 106, 105, 98, 100, 97, 96, 89, 87, 88, 92, 91, 90, 93,109, 103, 102, 108, 110, 117, 123, 140, 152, 163, 167, 186, 190, 191, 199, 203, 204, 198, 194, 192, 187, 157, 142, 150, 156, 164, 176, 182, 195, 177, 213, 225, 238, 253, 264, 252, 251, 231, 236, 250, 249, 248, 261, 262, 263, 286, 285, 284, 295, 307, 302, 299, 310, 323, 318, 330, 343, 335, 350, 362, 354, 366, 379, 375, 387, 393, 378, 365, 353, 338, 327, 314, 301, 273, 269, 274, 287, 306, 315, 319, 331, 346, 360, 372, 383, 400, 420, 426, 489, 427, 394, 347, 371, 334, 361, 490, 491, 492, 483, 484, 486, 487, 481, 478, 475, 474, 473, 476, 477, 479, 480, 472, 482, 449, 448, 447, 446, 445, 433, 434, 435, 436, 437, 438, 450, 451, 439, 440, 452, 453, 454, 441, 455, 485, 488, 429, 425, 409, 419, 418, 401, 386, 402, 403, 417, 416, 415, 404, 405, 414, 406, 407, 413, 412, 411, 399, 382, 373, 359, 345, 332, 320, 305, 288, 278, 289, 290, 279, 280, 281, 291, 292, 312, 321, 328, 308, 313, 326, 337, 352, 364, 377, 392, 385, 374, 358, 355, 339, 311, 294, 293, 283, 282, 260, 259, 247, 246, 245, 258, 257, 256, 255, 241, 242, 243, 244, 237, 232, 226, 218, 220, 219, 206, 202, 201, 183, 168, 171, 181, 196, 197, 214, 217, 210, 205, 207, 211, 216, 212, 227, 228, 233, 222, 230, 240, 266, 270, 272, 277, 298, 303, 317, 329, 342, 356, 367, 380, 395, 424, 432, 443, 444, 458, 459, 471, 470, 469, 468, 467, 466, 464, 462, 465, 463, 461, 456, 457, 460, 442, 430, 431, 423, 408, 388, 376, 363, 357, 348, 349, 336, 322, 309, 316, 324, 340, 351, 370, 369, 368, 389, 396, 410, 397, 390, 381, 333, 341, 344, 325, 296, 275, 271, 235, 229, 221, 208, 215, 209, 224, 239, 254, 267, 297, 276, 304, 384, 391, 398, 422, 421, 428, 24, 25, 26, 27, 28, 29, 30, 31, 33, 32, 23, 22, 21, 300, 268, 265, 234, 223, 20, 41, 44, 46, 45, 42, 43, 47, 49, 48, 51, 50, 40, 34, 35, 36, 37, 38, 39, 2] 
X=read_data(DATA_FILE)
X=np.array(X)
#path= [1,	98,	103,	82,	95,	107,	5,	100,	143,	97,	146,	26,	75,	18,	142,	85,	65,	132,	137,	50,	55,	58,	141,	83,	56,	90,	46,	92,	54,	138,	134,	131,	32,	23,	38,	67,	43,	109,	51,	20,	25,	110,	81,	29,	86,	135,	70,	108,	102,	114,	99,	19,	2,	37,	6,	28,	9,	42,	120,	47,	139,	40,	53,	118,	24,	12,	116,	101,	41,	57,	39,	127,	69,	36,	61,	11,	148,	130,	17,	66,	60,	140,	117,	129,	27,	31,	123,	74,	13,	106,	91,	119,	68,	128,	45,	71,	44,	64,	112,	136,	145,	144,	49,	147,	72,	80,	14,	122,	77,	133,	15,	78,	21,	150,	115,	4,	104,	22,	125,	149,	62,	3,	113,	10,	94,	88,	121,	79,	59,	16,	111,	105,	33,	126,	52,	93,	124,	35,	96,	89,	8,	7,	84,	30,	63,	48,	73,	76,	34,	87]
#path=[i-1 for i in path]
path=  [0, 1, 3, 4, 5, 8, 9, 10, 6, 7, 11, 18, 17, 16, 15, 14, 13, 12, 52, 56, 55, 53, 54, 64, 57, 58, 59, 63, 86, 115, 85, 84, 83, 62, 61, 60, 19, 65, 66, 68, 67, 69, 71, 70, 72, 73, 77, 80, 79, 82, 81, 95, 78, 76, 75, 74, 89, 87, 88, 90, 93, 91, 92, 94, 99, 100, 97, 96, 98, 105, 106, 116, 118, 112, 111, 107, 104, 101, 109, 113, 114, 119, 122, 126, 130, 132, 133, 129, 137, 139, 141, 143, 134, 136, 135, 145,165, 166, 161, 159, 158, 153, 151, 155,146, 147, 149, 144, 131, 128, 124, 125, 120, 121, 127, 138, 148, 154, 160, 162, 169, 172, 170, 174, 180, 189, 20, 223, 234, 265, 268, 300, 304, 276, 297, 267, 254, 239, 224, 209, 215, 208, 200, 175, 173, 179, 193, 185, 188, 184, 178, 168, 171, 181, 183, 201, 202, 203, 199, 204, 198, 194, 192, 187, 191, 190, 186, 167, 163, 152, 140, 123, 117, 110, 103, 102, 108, 142, 150, 156, 157, 164, 176, 182, 195, 177, 213, 225, 238, 253, 264, 252, 251, 231, 236, 250, 249, 248, 261, 262, 263, 286, 285, 295, 307, 302, 299, 310, 323, 318, 330, 343, 335, 350, 362, 354, 366, 379, 375, 387, 409, 419, 418, 401, 386, 402, 417, 416, 403, 404, 405, 415, 414, 413, 406, 407, 412, 411, 399, 382, 373, 359, 345, 332, 320, 305, 288, 278, 279, 289, 290, 280, 281, 291, 292, 312, 321, 328, 308, 313, 326, 337, 352, 364, 377, 392, 385, 374, 358, 355, 339, 311, 294, 284, 283, 293, 282, 260, 259, 247, 246, 245, 258, 257, 256, 255, 241, 242, 243, 244, 237, 232, 226, 206, 219, 220, 218, 212, 216, 211, 227, 228, 233, 222, 217, 210, 207, 205, 196, 197, 214, 221, 229, 235, 230, 240, 266, 270, 272, 277, 298, 309, 303, 317, 322, 336, 329, 342, 356, 367, 380, 395, 408, 388, 376, 363, 349, 348, 357, 340, 351, 344, 325, 324, 316, 296, 275, 271, 333, 341, 381, 390, 397, 368, 369, 370, 389, 396, 410, 457, 460, 442, 430, 431, 423, 424, 432, 443, 459, 458, 444, 445, 446, 433, 434, 435, 447, 448, 436, 437, 449, 482, 450, 451, 438, 439, 440, 452, 453, 454, 441, 455, 485, 488, 429, 425, 420, 400, 393, 378, 365, 353, 338, 327, 314, 301, 273, 269, 274, 287, 306, 315, 319, 331, 346, 360, 372, 383, 394, 347, 371, 334, 361, 490, 491, 492, 483, 484, 486, 487, 427, 426, 489, 481, 478, 475, 474, 473, 476, 477, 479, 480, 472, 471, 470, 469, 468, 467, 466, 464, 462, 465, 463, 461, 456, 24, 428, 421, 422, 384, 391, 398, 25, 26, 27, 28, 29, 30, 31, 38, 39, 36, 37, 35, 34, 33, 32, 23, 22, 21, 40, 41, 44, 46, 45, 42, 43, 49, 48, 47, 50, 51, 2]
path= [0, 1, 3, 4, 5, 8, 9, 10, 6, 7, 11, 18, 17, 16, 15, 14, 13, 12, 52, 56, 55, 53, 54, 64, 57, 58, 59, 63, 86, 115, 85, 84, 83, 62, 61, 60, 19, 65, 66, 68, 67, 69, 71, 70, 72, 73, 77, 80, 79, 82, 81, 95, 78, 76, 75, 74, 89, 87, 88, 90, 93, 91, 92, 94, 99, 100, 97, 96, 98, 105, 106, 116, 118, 112, 111, 107, 104, 101, 109, 113, 114, 119, 122, 126, 130, 132, 133, 129, 137, 139, 141, 143, 134, 136, 135, 145, 146, 147, 149, 165, 166, 161, 159, 155, 158, 153, 151, 144, 131, 128, 124, 125, 120, 121, 127, 138, 148, 154, 160, 162, 169, 172, 170, 174, 180, 189, 20, 223, 234, 265, 268, 300, 304, 276, 297, 267, 254, 239, 224, 209, 215, 208, 200, 175, 173, 179, 193, 185, 188, 184, 178, 171, 168, 181, 183, 201, 202, 206, 203, 199, 204, 198, 194, 192, 187, 191, 190, 186, 167, 163, 152, 140, 123, 117, 110, 103, 102, 108, 142, 150, 156, 157, 164, 176, 182, 195, 177, 213, 225, 238, 253, 264, 252, 251, 231, 236, 250, 249, 248, 261, 262, 263, 286, 285, 295, 307, 302, 299, 310, 323, 318, 330, 335, 350, 343, 354, 362, 375, 366, 379, 387, 409, 419, 401, 386, 402, 418, 417, 416, 403, 404, 405, 415, 414, 413, 406, 407, 412, 411, 399, 382, 373, 359, 345, 332, 320, 305, 288, 278, 279, 289, 290, 280, 281, 291, 292, 312, 321, 328, 308, 313, 326, 337, 352, 364, 377, 392, 385, 374, 358, 355, 339, 311, 294, 284, 283, 293, 282, 260, 259, 247, 246, 245, 258, 257, 256, 255, 241, 242, 243, 244, 237, 232, 226, 220, 219, 218, 212, 216, 211, 227, 228, 233, 222, 217, 210, 207, 205, 196, 197, 214, 221, 229, 235, 230, 240, 266, 270, 272, 277, 298, 303, 317, 309, 322, 336, 329, 342, 356, 367, 380, 395, 408, 388, 376, 363, 349, 348, 357, 340, 351, 344, 325, 324, 316, 296, 275, 271, 333, 341, 381, 390, 397, 368, 369, 370, 389, 396, 410, 457, 460, 442, 430, 431, 423, 424, 432, 443, 459, 458, 444, 445, 446, 433, 434, 435, 447, 448, 436, 437, 449, 482, 450, 451, 438, 439, 440, 452, 453, 454, 441, 455, 485, 488, 429, 425, 420, 400, 393, 378, 365, 353, 338, 327, 314, 301, 273, 269, 274, 287, 306, 315, 319, 331, 346, 360, 372, 383, 394, 347, 371, 334, 361, 490, 491, 492, 483, 484, 486, 487, 427, 426, 489, 481, 478, 475, 474, 473, 476, 477, 479, 480, 472, 471, 470, 469, 468, 467, 466, 464, 462, 465, 463, 461, 456, 24, 428, 421, 422, 384, 391, 398, 25, 26, 27, 28, 29, 30, 31, 38, 39, 36, 37, 35, 34, 33, 32, 23, 22, 21, 40, 41, 44, 46, 45, 42, 43, 49, 48, 47, 50, 51, 2]
path=   [0, 2, 51, 50, 47, 48, 49, 43, 42, 45, 46, 44, 41, 40, 21, 22, 23, 32, 33, 34, 35, 37, 36, 39, 38, 31, 30, 29, 28, 27, 26, 25, 398, 391, 384, 422, 421, 428, 24, 456, 461, 463, 465, 462, 464, 466, 467, 468, 469, 470, 471, 472, 480, 479, 477, 476, 473, 474, 475, 478, 481, 489, 426, 427, 487, 486, 484, 483, 492, 491, 490, 361, 334, 371, 347, 394, 383, 372, 360, 346, 331, 319, 315, 306, 287, 274, 269, 273, 301, 314, 327, 338, 353, 365, 378, 393, 400, 420, 425, 429, 488, 485, 455, 441, 454, 453, 452, 440, 439, 438, 451, 450, 482, 449, 437, 436, 448, 447, 435, 434, 433, 446, 445, 444, 458, 459, 443, 432, 424, 423, 431, 430, 442, 460, 457, 410, 396, 389, 370, 369, 368, 397, 390, 381, 333, 341, 325, 344, 351, 340, 357, 348, 349, 363, 376, 388, 408, 395, 380, 367, 356, 342, 336, 322, 329, 317, 303, 298, 309, 316, 324, 296, 275, 271, 235, 229, 221, 214, 217, 222,219, 220, 218, 212, 216, 211, 207, 210, 205,230, 240, 266, 270, 277, 272, 255, 242, 241, 233, 228, 227, 226, 232, 237, 244, 243, 256, 257, 258, 245, 246, 247, 259, 260, 282, 293, 283, 284, 294, 311, 339, 355, 358, 374, 385, 392, 377, 364, 352, 337, 326, 313, 308, 328, 321, 312, 292, 291, 281, 280, 290, 289, 279, 278, 288, 305, 320, 332, 345, 359, 373, 382, 399, 411, 412, 413, 407, 406, 405, 414, 415, 416, 404, 403, 417, 418, 402, 386, 401, 419, 409, 387, 375, 379, 366, 354, 362, 350, 335, 343, 330, 318, 323, 310, 299, 302, 307, 295, 285, 286, 263, 262, 261, 248, 249, 250, 236, 231, 251, 252, 264, 253, 238, 225, 213, 177, 195, 182, 176, 164, 157, 156, 150, 142, 108, 102, 103, 110, 117, 123, 140, 152, 163, 167, 186, 190, 191, 187, 192, 194, 198, 204, 199, 203, 206, 201, 202, 183, 168, 171, 181, 196, 197, 178, 184, 188, 185, 193, 179, 173, 175, 200, 208, 215, 209, 224, 239, 254, 267, 297, 276, 304, 300, 268, 265, 234, 223, 20, 189, 180, 174, 170, 172, 169, 162, 160, 154, 148, 138, 127, 121, 120, 125, 124, 128, 131, 144, 151, 153, 158, 159, 161, 166, 165, 155, 145, 146, 147, 149, 143, 141, 139, 137, 129, 133, 132, 130, 134, 136, 135, 126, 122, 119, 114, 113, 109, 101, 104, 107, 111, 112, 118, 116, 106, 105, 98, 96, 97, 100, 99, 94, 92, 91, 93, 90, 88, 87, 89, 74, 75, 76, 78, 95, 81, 82, 79, 80, 77, 73, 72, 70, 71, 69, 67, 68, 66, 65, 19, 60, 61, 62, 83, 84, 85, 115, 86, 63, 59, 58, 57, 64, 54, 53, 55, 56, 52, 12, 13, 14, 15, 16, 17, 18, 11, 7, 6, 10, 9, 8, 5, 4, 3, 1]
path=   [0, 84, 144, 190, 26, 197, 122, 14, 12, 78, 159, 161, 63, 19, 134, 41, 54, 66, 176, 64, 79, 160, 124, 180, 1, 34, 167, 184, 61, 82, 71, 129, 128, 133, 21, 74, 53, 150, 186, 5, 108, 106, 156, 30, 46, 119, 185, 126, 111, 154, 182, 7, 16, 24, 33, 89, 142, 145, 102, 113, 97, 57, 140, 170, 199, 87, 147, 27, 38, 37, 70, 55, 151, 177, 195, 4, 104, 42, 136, 132, 175, 112, 194, 181, 85, 138, 49, 94, 93, 90, 149, 143, 75, 69, 163, 139, 20, 153, 101, 22, 172, 168, 67, 29, 76, 157, 192, 127, 59, 166, 40, 88, 58, 2, 72, 188, 130, 179, 155, 80, 96, 164, 165, 95, 125, 86, 51, 10, 83, 47, 103, 196, 44, 32, 99, 73, 56, 35, 141, 68, 107, 13, 191, 3, 100, 162, 92, 105, 148, 189, 91, 9, 174, 98, 18, 118, 65, 152, 43, 187, 115, 169, 121, 193, 50, 62, 15, 117, 178, 17, 48, 109, 28, 183, 36, 123, 137, 8, 77, 81, 6, 198, 25, 60, 135, 31, 23, 158, 173, 120, 171, 45, 11, 146, 39, 131, 110, 116, 114, 52]
cond=sous_m([i for i in range(len(X))],len(X),X)
path= tsp_2_opt(np.array(cond), path) 
path=tsp_3_opt(np.array(cond), path)
#path= tsp_2_opt(np.array(cond), path)
#dist = tsp.TSP(X)
#path = dist.solve()
#print(path)


  

print(route_cost(np.array(cond),path),path)

import matplotlib.pyplot as plt
for i in range(len(X)):
            plt.plot(X[i][0] ,X[i][1], color='r', marker='o')
            #plt.text(X[i][0], X[i][1], i) 
            path.append(path[0])
            #path4.append(path4[0])
            
            x_points = [X[i][0] for i in path]
            y_points = [X[i][1] for i in path]
            #xx_points = [X[i][0] for i in path2]
            #yy_points = [X[i][1] for i in path2]

            
plt.plot(x_points, y_points, linestyle='--', color='b')
#plt.plot(xx_points, yy_points, linestyle='--', color='b')

plt.show()  # path visualization

