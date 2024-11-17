import pygad
import numpy as np
import random
import matplotlib.pyplot as plt
import time

# Opção para inserir o conjunto
print("Digite:")
print("0 - Para inserir manualmente um conjunto")
print("1 - Para uma criação automática do conjunto")
select = int(input("Escolha: "))

if select == 0:
    # Inserção manual
    input_numbers = input("Digite os números do conjunto separados por espaço: ")
    numbers = list(map(int, input_numbers.split()))
elif select == 1:
    # Geração automática
    tamanho_lista = int(input("Digite o tamanho do conjunto: (int value): "))
    limite_inferior = 1
    limite_superior = 100
    numbers = [random.randint(limite_inferior, limite_superior) for _ in range(tamanho_lista)]
else:
    print("Opção inválida. Encerrando.")
    exit()

# Função de fitness
def fitness_func(ga_instance, solution, solution_idx):
    subset_a = np.array(numbers)[solution == 1]
    subset_b = np.array(numbers)[solution == 0]
    sum_diff = abs(np.sum(subset_a) - np.sum(subset_b))
    fitness = 1.0 / (1.0 + sum_diff * 0.05)
    return fitness

# Função de mutação personalizada
def custom_mutation(offspring, ga_instance):
    for chromosome_idx in range(offspring.shape[0]):
        mutation_idx = np.random.randint(0, offspring.shape[1])
        offspring[chromosome_idx, mutation_idx] = 1 - offspring[chromosome_idx, mutation_idx]
    return offspring

# Parâmetros do algoritmo genético
num_generations = min(max(50, int(len(numbers)*(40/100))), 1000)
num_parents_mating = 5
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
                       gene_type=int,
                       stop_criteria=["reach_1"])  

start_time = time.time()

# Executa o algoritmo
ga_instance.run()

end_time = time.time()
execution_time = end_time - start_time


# Solução e resultados
solution, solution_fitness, solution_idx = ga_instance.best_solution()
subset_a = np.array(numbers)[solution == 1]
subset_b = np.array(numbers)[solution == 0]

np.set_printoptions(threshold=np.inf)


print("\n\n\n\n\n")
print(f"Conjunto S: {np.array(numbers)}\n")
print(f"Conjunto A: {subset_a}\n")
print(f"Conjunto B: {subset_b}\n")
print("\n")

print("Melhor solução encontrada:")
print(solution)
print("\n")
print("Soma do Subconjunto A:", np.sum(subset_a))
print("Soma do Subconjunto B:", np.sum(subset_b))
print(f"Fitness da melhor solução: {solution_fitness:.4f}")
print(f"Numero máximo de Gerações: {num_generations}")
print(f"Melhor valor de fitness encontrado apos {ga_instance.best_solution_generation} geracoes.")
print(f"Tempo para encontrar a melhor solução: {execution_time:.8f} segundos")
print("\n")
print("Operação encerrada")
print("\n")

# Plot da convergência
ga_instance.plot_fitness()

