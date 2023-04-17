#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 21:29:44 2023

@author: manuelperrin
"""

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



#Creating matrix containing differents sites and their characteristics then sort by ascending cleaning time

Sites = [S1, S2, S3, S4, S5, S6]

def sort_function(n):
  return n[2]

Sites.sort(key=sort_function)


#Creating matrix containing differents machines and their characteristics

Machines = [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, B1, B2, B3, B4, B5, C1, C2, C3, C4, C5, D1, D2]


#Creating matrix containing all the machines distinctly

Distinct_machines = []

while len(Machines)>0:
    
    while Machines[0][1]>0:
        
        Distinct_machines.append(Machines[0])
        Machines[0][1]-=1
    
    del Machines[0]
    
    
#Creating loop in order to select the best solution (with less machines) out of n possibilities (here n=100)

for a in range(100):
    
    #Starting with all the machines we dispose
    
    Start = Distinct_machines[:] 
    
    #Creating lists where we are going to store the temporary solutions (possibilities) for each site
    
    Site1 = []
    Site2 = []
    Site3 = [] 
    Site4 = []
    Site5 = []
    Site6 = []
    
    #Creating a list where we are going to store the final solution (initializing with all machines)
    
    Final_Solution = Start[:]
    
    #Creating a list containing a solution for each try (n=100)
    
    Final = []
    
    
    #Creating a loop to find a solution for each site
    
    for b in range(len(Sites)):
        
        #Storing cleaning area and time for site b
        
        cleaning_area = Sites[b][1]
        cleaning_time = Sites[b][2]
        
        #Creating a list to store the machine we are going to use for site b
        
        List=[]
        
        
        #Creating a loop to add to the list only machines for which the cleaning time is compatible with site b
        
        for c in range(len(Start)):
            if Start[c][2]<=cleaning_time:
                List.append(Start[c])
         
                
        #Creating a loop to take in account charging time possibility
        
        for d in range(len(List)):
            if List[d][2]*4.5<=cleaning_time:
                List[d][2]=List[d][2]*2
        
        #Creating a list to store the possible solutions for site b
        
        Solution = List[:]
        
        #Creating a loop to try m possible solutions (brut force, here m=1000)
        
        for e in range(1000):
            
            #Creating a list which will contain one by one the possible solutions
            
            Inter = []
            
            #Creating a list containg the machines in a randomize order
            
            Random = List[:]
            random.shuffle(Random)
            
            #Initializing variables space and incremental integer k
            
            space = 0
            k=0
            
            
            #Creating a loop adding machines will the space cleaned is inferior to the cleaning area needed
            
            while space < cleaning_area :
                Inter.append(Random[k][0])
                space = space + Random[k][3]
                k=k+1
                
            #Replacing the current solution by temporary solution only if the temporary solution uses less machines than current solution
                
            if len(Inter) < len(Solution):
                Solution = Inter
                
        #Attributing machines to the good site to retrieve in the output of the algorithm
        
        if b==0:
            Site1 = Solution
        
        if b==1:
            Site2 = Solution
            
        if b==2:
            Site3 = Solution
            
        if b==3:
            Site4 = Solution
            
        if b==4:
            Site5 = Solution
            
        if b==5:
            Site6 = Solution
            
        
        #Creating a list to retrieve the indexes of machines which will be used for site b
        
        Index = []
        
        
        #Creating a loop to add indexes to the list
        
        for f in range(len(Solution)):
            for g in range(len(Start)):
                if Solution[f] == Start[g][0]:
                    Index.append(g)
                    break
                
        #Sorting the list by descending order to delete the good indexes
                    
        Index.sort(reverse=True)
        
        
        #Creating a loop in order to delete the machines already used to the list of machines we dispose
        
        for h in range(len(Index)):
            del Start[Index[h]]
            
        #Adding the machines to solution list
        
        Final = Final + Solution
        
    #Replacing the current final solution by new solution only if the new solution uses less machines than current final solution
    
    if len(Final)<len(Final_Solution):
        Final_Solution = Final
        Final_Site1 = Site1
        Final_Site2 = Site2
        Final_Site3 = Site3
        Final_Site4 = Site4
        Final_Site5 = Site5
        Final_Site6 = Site6
        
#Displaying the final solution
        
print("Machines to be used are : ", Final_Solution)
print("Machines to be used for", Sites[0][0], "are :", Final_Site1)
print("Machines to be used for", Sites[1][0], "are :", Final_Site2)
print("Machines to be used for", Sites[2][0], "are :", Final_Site3)
print("Machines to be used for", Sites[3][0], "are :", Final_Site4)
print("Machines to be used for", Sites[4][0], "are :", Final_Site5)
print("Machines to be used for", Sites[5][0], "are :", Final_Site6)

print("Number of machines used :", len(Final_Solution))

print("Machines unused :", Start)
