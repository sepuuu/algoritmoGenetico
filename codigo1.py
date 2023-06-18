import numpy as np
import random
import matplotlib.pyplot as plt

N = 25 # tamaño del tablero
M = 50 # número de genes

# Matriz del tablero
matriz = np.zeros((N, N))

# Inicializa genes en posiciones aleatorias
genes = []
for _ in range(M):
    gen = [random.choice(range(8)) for _ in range(8)]
    genes.append((gen, (0, random.choice(range(N)))))

# Función para mover un gen de acuerdo a su ADN
def mover(gen, pos):
    movs = [(0, 1), (0, -1), (-1, 0), (1, 0), (-1, 1), (1, 1), (-1, -1), (1, -1)]
    x, y = pos
    for i in range(20):  # limite de pasos
        g = gen[i%8]    # repite los pasos en el gen después de 8
        dx, dy = movs[g]
        nx, ny = x + dx, y + dy
        if nx < 0 or nx >= N or ny < 0 or ny >= N: continue
        x, y = nx, ny
    return (x, y)

# Cruce de genes con selección de los mejores
def cruce(genes):
    genes.sort(key=lambda x: x[1][0], reverse=True)  # ordena por la distancia recorrida hacia abajo
    sobrevivientes = genes[:M//2]  # toma la mitad superior
    nueva_gen = []
    for _ in range(M):
        gen1, gen2 = random.sample(sobrevivientes, 2)
        punto = random.choice(range(8))
        hijo = gen1[0][:punto] + gen2[0][punto:]
        nueva_gen.append((hijo, (0, random.choice(range(N)))))
    return nueva_gen

# Visualización de los datos con matplotlib
def graficar(genes):
    tablero = np.zeros((N, N))
    for gen, pos in genes:
        tablero[pos] = 1
    plt.imshow(tablero, cmap='gray')
    plt.show()

# Ciclo principal
for generacion in range(20):
    print(f"Generación {generacion}")
    genes = [(gen, mover(gen, pos)) for gen, pos in genes]
    graficar(genes)
    genes = cruce(genes)
    
#evaluar los genes y guardar el mejor
#normalizar el movimiento
#