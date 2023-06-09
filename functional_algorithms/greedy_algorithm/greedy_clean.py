# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 15:52:13 2023
@author: egonzalez
"""

import pandas

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
Sites = [S1, S2, S3, S4, S5, S6]

#Add productivity on running time ratio column

for machine in Machines:
    
    ratio = round(machine[3] / machine[2], 5)
    
    machine.append(ratio)


#Creating matrix containing differents machines and their characteristics then sort by descending productivity on running time ratio

def machine_sort(n):
    
  return n[4],n[3]

Machines.sort(reverse=True, key=machine_sort)





#Creating matrix containing differents sites and their characteristics then sort by ascending cleaning time

def site_sort(n):
    
  return n[2]

Sites.sort(key=site_sort)



#Creating matrix containing all the machines distinctly

Distinct_machines = []

while len(Machines)>0:
    
    while Machines[0][1]>0:
        
        Distinct_machines.append(Machines[0])
        
        Machines[0][1]-=1
    
    del Machines[0]
    

#Cleaning the matrix of its empty column

Distinct_machines = [[item[0],item[2], item[3], item[4]] for item in Distinct_machines]



# Create empty dictionary with site names as keys

selected_machines_by_site = {site[0]: [] for site in Sites}




for site in Sites: # iterate through each site in the list of "Sites"
    
    temp_list = [] # initialize an empty list to store selected machines for this site
    
    surface = site[1] # set "surface" equal to the surface area of the site
    
    machines_to_remove = [] # initialize an empty list to store machines that have been selected for removal
    
    for i, machine in enumerate(Distinct_machines): # iterate through each machine in the list of "Distinct_machines"
        
        if machine[1] <= site[2]: # check if the running time of the machine is less than or equal to the maximum running for this site
            
            if surface <= 0: # check if there is any remaining surface area left to clean
                
                break  # if there is no remaining surface area, stop selecting machines for this site
            
            else:
                
                temp_list.append(machine) # add the machine to the list of selected machines for this site
            
                surface -= machine[2] # decrease the remaining surface area by the cleaning area of the machine
            
                del Distinct_machines[i] # remove the selected machines from the "Distinct_machines" list
            
    for temp in temp_list: # add all the selected machines for this site to the dictionary "selected_machines_by_site" with the site ID as the key
        
        selected_machines_by_site[site[0]].append(temp)


        
for site, machines in selected_machines_by_site.items():
    print(f"{site}: {machines}")
    total = sum(machine[2] for machine in machines)
    print(f"Total selected machines for {site}: {total}")
    print(f"Number of machines used for {site}: {len(machines)}\n") # print the total size of the machines selected for this site and the number of machines used at this site
    
    

print("Distinct_machines\n",Distinct_machines, "\n", "len(Distinct_machines) = ", len(Distinct_machines))
