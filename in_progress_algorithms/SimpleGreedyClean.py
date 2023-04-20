# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 15:52:13 2023

@author: egonzalez
"""

import pandas
import copy
import random


#Machines

A1 = ['A1', 4, 1, 1225]
A2 = ['A2', 5, 1.5, 1575]
A3 = ['A3', 1, 2.5, 2475]
A4 = ['A4', 7, 2.5, 1750]
A5 = ['A5', 3, 3, 1750]
A6 = ['A6', 10, 3.5, 3150]
A7 = ['A7', 7, 2.5, 2700]
A8 = ['A8', 1, 2.5, 3150]
A9 = ['A9', 9, 3.5, 3150]
A10 = ['A10', 10, 4, 3825]
B1 = ['B1', 6, 1.5, 1400]
B2 = ['B2', 8, 2.5, 1720]
B3 = ['B3', 6, 3.5, 1720]
B4 = ['B4', 10, 3.5, 2200]
B5 = ['B5', 8, 5, 2200]
C1 = ['C1', 1, 8, 5600]
C2 = ['C2', 4, 3, 2970]
C3 = ['C3', 10, 3.5, 2460]
C4 = ['C4', 5, 4.5, 7740]
C5 = ['C5', 2, 5, 9000]
D1 = ['D1', 7, 3, 7100]
D2 = ['D2', 3, 6, 4250]

#Sites

S1 = ['S1', 14300, 5]
S2 = ['S2', 82000, 4]
S3 = ['S3', 87000, 4]
S4 = ['S4', 21000, 3]
S5 = ['S5', 27900, 8]
S6 = ['S6', 61000, 4]



Machines = [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, B1, B2, B3, B4, B5, C1, C2, C3, C4, C5, D1, D2]

#Add productivity on running time ratio column

for machine in Machines:
    ratio = round(machine[3] / machine[2], 5)
    machine.append(ratio)


#Creating matrix containing differents machines and their characteristics then sort by descending productivity on running time ratio

def machine_sort(n):
  return n[4],n[3]

Machines.sort(reverse=True, key=machine_sort)


Sites = [S1, S2, S3, S4, S5, S6]


#Creating matrix containing differents sites and their characteristics then sort by ascending cleaning time

def site_sort(n):
  return n[2]

Sites.sort(key=site_sort)

#print(Sites)

#Creating matrix containing all the machines distinctly


Distinct_machines = []

while len(Machines)>0:
    
    while Machines[0][1]>0:
        
        Distinct_machines.append(Machines[0])
        Machines[0][1]-=1
    
    del Machines[0]
    

Distinct_machines = [[item[0],item[2], item[3], item[4]] for item in Distinct_machines]

#print(len(Distinct_machines))


# Create empty dictionary with site names as keys
selected_machines_by_site = {site[0]: [] for site in Sites}

for site in Sites:
    
    temp_list = []
    
    surface = site[1]
    
    machines_to_remove = []
    
    for i, machine in enumerate(Distinct_machines):
        
        if machine[1] <= site[2]:
            
            if surface <= 0:
                
                break
            
            else:
                
                temp_list.append(machine)
            
                surface -= machine[2]
            
                del Distinct_machines[i]
            
    for temp in temp_list:
        
        selected_machines_by_site[site[0]].append(temp)

#print(selected_machines_by_site)

"""   
for site, machines in selected_machines_by_site.items():
    print(f"{site}: {machines}")
    total = sum(machine[2] for machine in machines)
    print(f"Total selected machines for {site}: {total}")
    print(f"Number of machines used for {site}: {len(machines)}\n")
    
    

print("Distinct_machines\n",Distinct_machines, "\n", "len(Distinct_machines) = ", len(Distinct_machines))
"""

# Define the neighborhood structure
def generate_neighbors(solution, distinct_solution):
    
    neighbors = []
    
    distinct_list = []
    
    for s in range(len(solution)):
        
        for i in range(len(solution[s][1])):
            
            for j in range(i+1, len(solution[s][1])):
                
                distinct = copy.deepcopy(distinct_solution)
                
                neighbor = copy.deepcopy(solution)
                
                r1 = random.choice(distinct)
                
                i1 = distinct.index(r1)
                
                distinct[i1], neighbor[s][1][i] = neighbor[s][1][i], distinct[i1]
                
                r2 = random.choice(distinct)
                
                i2 = distinct.index(r2)
                
                distinct[i2], neighbor[s][1][j] = neighbor[s][1][j], distinct[i2]
                
                for site in neighbor:
                    
                    site_name = site[0]
                    
                    cleaning_machines = sum([machines[2] for machines in site[1]])
                    
                    cleaning_surface = sum([sites[1] for sites in Sites if sites[0]==site_name])
                    
                    if cleaning_machines < cleaning_surface:
                        
                        break
                else:    
                                    
                    neighbors.append(neighbor)
                
                    distinct_list.append(distinct)

    return neighbors, distinct_list


site_list = list(selected_machines_by_site.items())


def less_machines(machine_list, distinct_list):
    
    for m in range(len(machine_list)):
            
        for s in range(len(machine_list[m])):
            
            site_name = machine_list[m][s][0]
            
            cleaning_surface = sum([sites[1] for sites in Sites if sites[0] == site_name])
            
            cleaning_machines = sum([machines[2] for machines in machine_list[m][s][1]])
                
            for i in range(len(machine_list[m][s][1])):
                    
                for j in range(i+1, len(machine_list[m][s][1])):
                    
                    if j==i:
                        
                        continue
                    
                    for d in range(len(distinct_list[m])):
                          
                        new_surface = cleaning_machines - machine_list[m][s][1][i][2] - machine_list[m][s][1][j][2] + distinct_list[m][d][2]
                        
                        if  new_surface > cleaning_surface:
                            
                        #print(site_name, "\n", new_surface,"\n", cleaning_machines, "\n", cleaning_surface)
                            
                            machine_list[m][s][1].append(distinct_list[m][d])
                            
                            distinct_list[m].append(machine_list[m][s][1][i])
                            
                            distinct_list[m].append(machine_list[m][s][1][j])
                            
                            machine_list[m][s][1].remove(machine_list[m][s][1][j])
                            
                            machine_list[m][s][1].remove(machine_list[m][s][1][i])
                            
                            break
                        
                        break
                    
                    break
                
                
                        
    return machine_list, distinct_list


"""
# Define the evaluation function
def evaluate(solution):
    
    total_machines = sum([sum([m[1] for m in job]) for job in solution])
    
    return total_machines

"""
"""
def tabu_search(initial_solution, max_iterations, tabu_list_size):
    current_solution = initial_solution
    best_solution = initial_solution
    tabu_list = []
    for i in range(max_iterations):
        current_neighbors = generate_neighbors(current_solution, Distinct_machines)[0]
         # Find the best neighbor that is not in the tabu list
        best_neighbor = None
        for neighbor in neighbors:
            if neighbor not in tabu_list:
                if best_neighbor is None or evaluate(neighbor) < evaluate(best_neighbor):
                    best_neighbor = neighbor
        # If all neighbors are in the tabu list, select a random one
        if best_neighbor is None:
            best_neighbor = random.choice(neighbors)
        # Update the tabu list
        tabu_list.append(best_neighbor)
        if len(tabu_list) > tabu_list_size:
            tabu_list.pop(0)
        # Update the current solution
        current_solution = best_neighbor
        # Update the best solution
        if evaluate(current_solution) < evaluate(best_solution):
            best_solution = current_solution
    return best_solution
        
"""