def assign_probabilities(num_individuals, p):
    probabilities = []
    for i in range(num_individuals):
        probability = p * (1 - p) ** i
        probabilities.append(probability)
    return probabilities

# ParÃ¡metros
num_individuals = 10
p = 0.5

# Asignar probabilidades
probabilities = assign_probabilities(num_individuals, p)

# Imprimir las probabilidades asignadas
for i, prob in enumerate(probabilities):
    print(f"Probabilidad para el individuo {i+1}: {prob}")