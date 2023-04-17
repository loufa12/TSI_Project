#!/usr/bin/env python
# coding: utf-8

# In[3]:


import random
import operator
import numpy as np
import matplotlib.pyplot as plt
from deap import creator, base, tools, algorithms

# Données des machines de nettoyage (nom, temps de fonctionnement en heures, productivité en m²)
scrubbers = [("A1", 1, 1225), ("A2", 1.5, 1575), ("A3", 2.5, 2475), ("A4", 2.5, 1750), 
             ("A5", 3, 1750), ("A6", 3.5, 3150), ("A7", 2.5, 2700), ("A8", 2.5, 3150), 
             ("A9", 3.5, 3150), ("A10", 4, 3825), ("B1", 1.5, 1400), ("B2", 2.5, 1720), 
             ("B3", 3.5, 1720), ("B4", 3.5, 2200), ("B5", 5, 2200), ("C1", 8, 5600), 
             ("C2", 3, 2970), ("C3", 3.5, 2460), ("C4", 4.5, 7740), ("C5", 5, 9000)]

# Données des sites de nettoyage (nom, surface de nettoyage en m², temps de nettoyage en heures)
sites = [("Site 1", 23900, 7), ("Site 2", 19200, 4), ("Site 3", 16400, 6), ("Site 4", 16200, 6), 
         ("Site 5", 33000, 3)]


# Fonction d'évaluation d'un individu
def evaluate(individual):
    #
    #Évalue la qualité d'un individu en calculant le temps total de nettoyage de tous les sites avec les machines 
    #attribuées selon cet individu.
    #
    # Création d'un dictionnaire qui contiendra la liste des machines attribuées à chaque site
    site_assignments = {site[0]: [] for site in sites}
    
    # Attribution des machines aux sites
    for machine, site_index in zip(individual, range(len(sites))):
        site_assignments[sites[site_index][0]].append(scrubbers[machine][0])
    
    # Calcul du temps total de nettoyage pour chaque site
    total_cleaning_time = 0
    for site in sites:
        cleaning_time = site[1] / max(scrubbers[machine][2] for machine in site_assignments[site[0]])
        total_cleaning_time += cleaning_time if cleaning_time > site[2] else site[2]
    
    return total_cleaning_time

# Initialisation de la classe Fitness pour maximiser la qualité des individus
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# Initialisation de la classe individu
creator.create("Individual", list, fitness=creator.FitnessMax)
               
class CleaningProblem:
    def __init__(self, scrubbers, sites):
        
        self.scrubbers = scrubbers
        
        # Création des listes de poids et de bornes
        self.weights = np.array([scrubber[1] for scrubber in scrubbers])
        self.bounds = [(0, site[1] / scrubbers[-1][2]) for site in sites]
        
        self.sites = sites
        self.num_scrubbers = len(scrubbers)
        self.num_sites = len(sites)

        
    
    def eval_solution(self, individual):
        # Calcul de la durée totale de nettoyage
        cleaning_times = [sum(self.weights[np.where(individual == i)[0]]) for i in range(self.num_sites)]
        total_time = max(cleaning_times)

        # Calcul de la distance parcourue par chaque machine de nettoyage
        distances = [0] * self.num_scrubbers
        for i in range(1, self.num_sites):
            for j in range(self.num_scrubbers):
                if individual[i-1] == j:
                    distances[j] += self.sites[i][1]

        # Calcul du coût total de la solution
        total_cost = sum([distance * scrubber[2] for distance, scrubber in zip(distances, self.scrubbers)])

        return total_time, total_cost

problem = CleaningProblem(scrubbers, sites)

POPULATION_SIZE = 400
P_CROSSOVER = 1
P_MUTATION = 0.05
MAX_GENERATIONS = 300
HALL_OF_FAME_SIZE = 10
RANDOM_SEED = 50
TOURNAMENT_SIZE = 3

random.seed(RANDOM_SEED)

creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)
toolbox = base.Toolbox()

toolbox.register("indexes", random.sample, range(problem.num_scrubbers), problem.num_sites)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indexes)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", problem.eval_solution)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=1.0/problem.num_sites)
toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("min", lambda x: (x[0], x[1]))

hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

population = toolbox.population(n=POPULATION_SIZE)
population, logbook = algorithms.eaSimple(population, toolbox, cxpb = P_CROSSOVER, mutpb = P_MUTATION, ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True)
# Création de la population initiale
population = toolbox.population(n=100)
cxpb = P_CROSSOVER
mutpb = P_MUTATION
ngen=MAX_GENERATIONS
# Évaluation de la population initiale
fitnesses = list(map(toolbox.evaluate, population))
for ind, fit in zip(population, fitnesses):
    ind.fitness.values = fit

# Affichage de la meilleure solution de la population initiale
best_ind = tools.selBest(population, k=1)[0]
print("Best solution : {}".format(best_ind))
print("Fitness of the best solution : {}".format(best_ind.fitness.values[0]))

    
# Lancement de l'algorithme génétique
for generation in range(50):
    print("Generation : {}".format(generation))
    
    # Sélection des parents
    parents = toolbox.select(population, len(population))
    
    # Clonage des parents
    offspring = list(map(toolbox.clone, parents))
    
    # Croisement des parents clonés
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < cxpb:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values
            
    # Mutation des enfants
    for mutant in offspring:
        if random.random() < mutpb:
            toolbox.mutate(mutant)
            del mutant.fitness.values
    
    # Évaluation des enfants avec une fitness invalide
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
        
    # Remplacement de la population par les enfants
    population[:] = offspring
    
    
    # Affichage de la meilleure solution de la génération actuelle
    best_ind = tools.selBest(population, k=1)[0]
    print("Best solution for generation : {}".format(best_ind))
    print("Fitness of the best solution : {}".format(best_ind.fitness.values[0]))
    
        
    # Génération donnée
    #generation = random.sample(range(0, len(scrubbers)), 5)
    #print("Generation:", generation)


    selected_column = [scrubber[2] for scrubber in scrubbers if scrubbers.index(scrubber) in best_ind]
    selected_time = [scrubber[1] for scrubber in scrubbers if scrubbers.index(scrubber) in best_ind]
    # Addition des valeurs sélectionnées
    total = sum(selected_column)
    time = max(selected_time)
    # Affichage du total
    print("Total area cleaned:", total,"m²")
    print("Total area time:", time ,"hours")
    
print("-- End of the evolution --")

best_ind = tools.selBest(population, k=1)[0]
print("Best solution Found : {}".format(best_ind))
print("Fitness of the best solution : {}".format(best_ind.fitness.values[0]))



# In[67]:


def calculate_total_time(individual):
    total_time = 0
    for site, scrubber_idx in zip(sites, individual):
        scrubber = scrubbers[scrubber_idx]
        total_time += site[2] / scrubber[1]  # Fix the calculation here
    return total_time
# Génération donnée
generation = random.sample(range(0, len(scrubbers)), 5)
print("Generation:", generation)


selected_column = [scrubber[2] for scrubber in scrubbers if scrubbers.index(scrubber) in generation]

# Addition des valeurs sélectionnées
total = sum(selected_column)

# Affichage du total
print("Total area cleaned:", total,"m²")
total_time_used = calculate_total_time(generation)
print("Total time used: {} hours".format(total_time_used))


# In[ ]:





# In[ ]:





# In[4]:


def evaluation_fitness(population, scrubbers, sites):
    fitness_scores = []
    for individual in population:
        total_productivity = 0
        for site_index, site in enumerate(sites):
            site_productivity = 0
            for scrubber_index, scrubber in enumerate(individual[site_index]):
                site_productivity += scrubbers[scrubber_index][2] * scrubber
            total_productivity += site_productivity
        fitness_scores.append(total_productivity)
    return fitness_scores


# La fonction evaluation_fitness prend en entrée la population d'individus (où chaque individu représente une affectation de machines de nettoyage aux sites de nettoyage), les données des machines de nettoyage (scrubbers) et les données des sites de nettoyage (sites).
# 
# Elle itère sur chaque individu dans la population et pour chaque site de nettoyage, elle calcule la productivité totale des machines de nettoyage attribuées à ce site en multipliant la productivité de chaque machine de nettoyage par le nombre d'instances de cette machine attribuées au site (représenté par la valeur dans l'individu pour ce site). Ensuite, elle ajoute la productivité totale du site à la productivité totale de tous les sites de nettoyage pour obtenir la qualité (fitness) de l'individu. Les scores de fitness sont stockés dans une liste fitness_scores et sont renvoyés à la fin de la fonction.

# In[5]:


import random

def generate_initial_population(pop_size, scrubbers, sites):
    population = []
    for _ in range(pop_size):
        individual = []
        for site in sites:
            site_cleaning_time = site[2]
            site_scrubbers = []
            while site_cleaning_time > 0:
                scrubber_index = random.randint(0, len(scrubbers)-1)
                scrubber_productivity = scrubbers[scrubber_index][2]
                max_scrubbers = site_cleaning_time // scrubber_productivity
                num_scrubbers = random.randint(0, max_scrubbers)
                site_scrubbers.append(num_scrubbers)
                site_cleaning_time -= num_scrubbers * scrubber_productivity
            individual.append(site_scrubbers)
        population.append(individual)
    return population


# La fonction generate_initial_population prend en entrée la taille de la population (pop_size), les données des machines de nettoyage (scrubbers) et les données des sites de nettoyage (sites).
# 
# Elle itère sur chaque individu dans la population et pour chaque site de nettoyage, elle génère un ensemble aléatoire de machines de nettoyage à attribuer à ce site en respectant les contraintes de temps de nettoyage. Pour cela, elle choisit aléatoirement un index de machine de nettoyage (scrubber_index), puis génère un nombre aléatoire de machines de nettoyage (num_scrubbers) à attribuer au site, en s'assurant que la somme de la productivité de ces machines ne dépasse pas le temps de nettoyage disponible pour le site. Les affectations de machines de nettoyage pour chaque site sont stockées dans une liste site_scrubbers, et cette liste est ajoutée à l'individu. Finalement, la population complète d'individus est renvoyée à la fin de la fonction.

# In[6]:



def assess_fitness(population):
    fitness_scores = []
    for individual in population:
        fitness = 0
        for i in range(len(individual)):
            site_scrubbers = individual[i]
            site_productivity = sum([scrubbers[j][2] * site_scrubbers[j] for j in range(len(site_scrubbers))])
            fitness += site_productivity
        fitness_scores.append(fitness)
    return fitness_scores

def selection(fitness_scores, num_parents):
    parents_indices = random.sample(range(len(fitness_scores)), num_parents)
    return parents_indices

def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1)-1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def mutation(individual, mutation_rate):
    mutated_individual = individual.copy()
    for i in range(len(mutated_individual)):
        for j in range(len(mutated_individual[i])):
            if random.random() < mutation_rate:
                mutated_individual[i][j] = random.randint(0, max_scrubbers)
    return mutated_individual

def genetic_algorithm(pop_size, num_generations, scrubbers, sites, mutation_rate):
    population = generate_initial_population(pop_size, scrubbers, sites)
    for generation in range(num_generations):
        fitness_scores = assess_fitness(population)
        parents_indices = selection(fitness_scores, pop_size//2)
        offspring = []
        for i in range(0, len(parents_indices), 2):
            parent1 = population[parents_indices[i]]
            parent2 = population[parents_indices[i+1]]
            child1, child2 = crossover(parent1, parent2)
            child1 = mutation(child1, mutation_rate)
            child2 = mutation(child2, mutation_rate)
            offspring.append(child1)
            offspring.append(child2)
        population += offspring
        fitness_scores += assess_fitness(offspring)
        sorted_indices = sorted(range(len(fitness_scores)), key=lambda k: fitness_scores[k], reverse=True)
        population = [population[i] for i in sorted_indices[:pop_size]]
    return population


# La fonction genetic_algorithm prend en entrée la taille de la population (pop_size), le nombre de générations à effectuer (num_generations), les données des machines de nettoyage (scrubbers), les données des sites de nettoyage (sites), et le taux de mutation (mutation_rate).
# 
# Elle commence par générer une population initiale à l'aide de la fonction generate_initial_population. Ensuite, elle boucle sur le nombre de générations spécifié. À chaque génération, elle évalue la fitness de la population avec la fonction assess_fitness, sélectionne les parents pour la reproduction avec la fonction selection, effectue le croisement des parents pour créer des descendants avec la fonction crossover, applique la mutation aux descendants avec la fonction mutation, ajoute les descendants à la population, évalue à nouveau la fitness de la population avec les descendants inclus, trie la population en fonction de la fitness et sélectionne les individus les mieux adaptés pour former la population de la génération suivante. Ce processus est répété pour le nombre de générations spécifié.
# 
# Veuillez noter que les fonctions generate_initial_population, assess_fitness, selection, crossover, et mutation doivent être implémentées en fonction de la structure spécifique de vos données et des contraintes de votre problème. Vous pouvez les adapter en conséquence pour correspondre à votre cas d'utilisation spécifique.
# 
# Assurez-vous également d'avoir les données nécessaires pour les machines de nettoyage (scrubbers) et les sites de nettoyage (sites) prêtes avant d'appeler la fonction genetic_algorithm. Vous pouvez les préparer dans un format approprié pour votre problème avant de les passer à la fonction.
# 
# N'oubliez pas de spécifier le critère d'arrêt approprié dans la boucle de génération, tel que le nombre maximum de générations à atteindre, ou un critère de convergence basé sur la fitness ou d'autres métriques spécifiques à votre problème.

# In[7]:


def get_best_solution(population, scrubbers, sites):
    best_fitness = 0
    best_solution = None
    for individual in population:
        fitness = 0
        for i in range(len(individual)):
            site_scrubbers = individual[i]
            site_productivity = sum([scrubbers[j][2] * site_scrubbers[j] for j in range(len(site_scrubbers))])
            fitness += site_productivity
        if fitness > best_fitness:
            best_fitness = fitness
            best_solution = individual
    return best_solution, best_fitness


# La fonction get_best_solution prend en entrée la population générée par l'algorithme génétique (population), les données des machines de nettoyage (scrubbers), et les données des sites de nettoyage (sites).
# 
# Elle évalue la fitness de chaque individu dans la population en calculant la productivité totale de toutes les machines de nettoyage attribuées à chaque site, et en additionnant ces productivités pour obtenir la fitness totale de l'individu. Elle maintient également une variable pour suivre la meilleure fitness trouvée et la meilleure solution associée.
# 
# Enfin, elle retourne la meilleure solution trouvée dans la population ainsi que sa fitness correspondante. Vous pouvez appeler cette fonction après avoir exécuté l'algorithme génétique pour obtenir la meilleure solution trouvée par l'algorithme.
