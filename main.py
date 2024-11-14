import pygad
import numpy as np
import random
 
tamanho_lista = 100
limite_inferior = 1
limite_superior = 100  
 
# Gera a lista de números aleatórios
numbers = [random.randint(limite_inferior, limite_superior) for _ in range(tamanho_lista)]
 
 
def fitness_func(ga_instance, solution, solution_idx):
 
    #sum_set = np.sum(numbers)
 
    # Subconjuntos baseados nos valores dos genes (0 ou 1)
    subset_a = np.array(numbers)[solution == 1]
    subset_b = np.array(numbers)[solution == 0]
    
    sum_diff = abs(np.sum(subset_a) - np.sum(subset_b))
 
    fitness = 1.0 / (1.0 + sum_diff*0.1)
    #fitness = 1.0 - sum_diff / sum_set
 
 
    return fitness
 
 
 
 
# Função de mutação personalizada que alterna entre 0 e 1
def custom_mutation(offspring, ga_instance):
    for chromosome_idx in range(offspring.shape[0]):
 
        # Escolhe um gene aleatoriamente para mutação
        mutation_idx = np.random.randint(0, offspring.shape[1])
 
        # Alterna entre 0 e 1
        offspring[chromosome_idx , mutation_idx] = 1 - offspring[chromosome_idx , mutation_idx]
 
    return offspring
 
 
 
 
# Parâmetros do algoritmo genético
num_generations = 100
num_parents_mating = 8
sol_per_pop = 10
num_genes = len(numbers)
 
 
 
 
 
# Configuração do algoritmo genético
ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_func,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       init_range_low=0,
                       init_range_high=1,
                       parent_selection_type="sss",
                       keep_parents=1,
                       crossover_type="single_point",
                       mutation_type=custom_mutation,  
                       mutation_percent_genes=10,
                       save_solutions=True,
                       save_best_solutions=True,
                       gene_type=int)  
 
 
ga_instance.run()
 
solution, solution_fitness, solution_idx = ga_instance.best_solution()
subset_a = np.array(numbers)[solution == 1]
subset_b = np.array(numbers)[solution == 0]
 
 
print(np.sum(subset_a))
print(np.sum(subset_b))
print(f"Fitness value of the best solution: {solution_fitness:.4f}")
 
ga_instance.plot_fitness()