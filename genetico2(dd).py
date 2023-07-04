import numpy as np
import matplotlib.pyplot as plt

plt.ion()

grida_dim = 20
grida = np.full((grida_dim, grida_dim, 3), 255)

movimientos = [(0, -1), (0, 1), (1, 0), (-1, 0), (1, -1), (-1, -1), (1, 1), (-1, 1), (0, 0)]

colores = [(128, 0, 0), (139, 0, 0), (165, 42, 42), (178, 34, 34), (220, 20, 60), (255, 0, 0), (255, 99, 71), (255, 165, 0), (255, 140, 0)]

num_individuos = 50
color_fondo = [255, 255, 255]

class Individuo:
    def __init__(self, x, y, probs, color):
        self.x = x
        self.y = y
        self.probs = probs
        self.color = color
        self.en_meta = False
        self.iteraciones = 0

    @property
    def fitness(self):
        return 1 / (1 + self.iteraciones)  # Fitness es inversamente proporcional al número de iteraciones

def cruce(padre, madre):
    punto_cruce = np.random.randint(1, len(padre.probs))  # Elegir un punto de cruce al azar
    # Hasta el punto de cruce, el hijo obtiene los genes del padre; después del punto de cruce, los genes son de la madre.
    prob_hijo = np.concatenate((padre.probs[:punto_cruce], madre.probs[punto_cruce:]))
    return prob_hijo

def generar_probabilidades_ruleta(n, p):
    probabilidades = [p * (1 - p) ** i for i in range(n)]
    return np.array(probabilidades) / np.sum(probabilidades)

seed = 1234
np.random.seed(seed)

# Inicializar individuos_en_meta fuera del bucle principal
individuos_en_meta = []
individuos = []

def seleccion_ruleta(individuos):
    fitnesses = np.array([ind.fitness for ind in individuos])
    fitnesses = fitnesses / np.sum(fitnesses)  # Normalizar los fitnesses para que sumen 1

    # Selecciona un individuo usando la ruleta
    return np.random.choice(individuos, p=fitnesses)

def mutacion(probs, tasa_mutacion=0.01):
    # Se aplica una mutación aleatoria a los genes con una pequeña probabilidad
    for i in range(len(probs)):
        if np.random.rand() < tasa_mutacion:
            probs[i] = np.random.rand()

    # Normalizar las probabilidades para que sumen 1 después de la mutación
    return probs / np.sum(probs)


def generar_hijos(individuos_en_meta, num_hijos, p):
    hijos = []
    probabilidades_ruleta = generar_probabilidades_ruleta(len(individuos_en_meta), p)
    
    for _ in range(num_hijos):
        padre = seleccion_ruleta(individuos_en_meta)
        madre = seleccion_ruleta(individuos_en_meta)

        while padre == madre:  # Asegurarse de que el padre y la madre sean diferentes
            madre = seleccion_ruleta(individuos_en_meta)

        hijo_probs = cruce(padre, madre)
        hijo_probs = mutacion(hijo_probs)  # Aplicar mutación a los genes del hijo
        movimiento_predominante = np.argmax(hijo_probs)
        hijo_color = colores[movimiento_predominante]

        # Los hijos comienzan en las primeras dos columnas del grid
        hijo = Individuo(np.random.randint(0, 2), np.random.randint(0, grida_dim), hijo_probs, hijo_color)
        hijos.append(hijo)

    return hijos

plot_interval = 50  # Mostrar el plot cada 5 generaciones
plot_counter = 0
num_iteraciones = 50

for gen in range(300):  # Número de generaciones
    grida = np.full((grida_dim, grida_dim, 3), 255, dtype=int)

    if individuos_en_meta:
        if len(individuos_en_meta) > 1:
            individuos = generar_hijos(individuos_en_meta, len(individuos_en_meta), p=0.7)
            individuos_en_meta = []

    for _ in range(num_individuos - len(individuos)):
        # Todos los individuos comienzan en las primeras dos columnas del grid
        x = np.random.randint(0, 2)
        y = np.random.randint(0, grida_dim)
        probs = np.random.rand(len(movimientos))
        probs = probs / np.sum(probs)
        movimiento_predominante = np.argmax(probs)
        individuo = Individuo(x, y, probs, colores[movimiento_predominante])
        individuos.append(individuo)
        grida[y, x] = individuo.color

    meta = grida_dim - 1

    fig, ax = plt.subplots()
    img = ax.imshow(grida, origin='lower')

    def visualizar_grida(grida, gen, iteracion):
        img.set_data(grida)
        if gen == 0 or (gen + 1) % plot_interval == 0:
            plt.imshow(grida, extent=[0, grida_dim, 0, grida_dim], origin='lower')
            plt.grid(color='gray', linestyle='--', linewidth=0.2)
            plt.title(f'Generación {gen + 1}, Iteración {iteracion + 1}')

            plt.draw()
            plt.pause(0.05)
        else:
            plt.close()

    if gen == 0 or (gen + 1) % plot_interval == 0:
        visualizar_grida(grida, gen, 0)
        plt.pause(0.01)

    if (gen + 1) % 5 == 0:
        #num_iteraciones -= 5
        num_iteraciones=num_iteraciones

    posiciones_ocupadas = set((ind.y, ind.x) for ind in individuos)

    for i in range(num_iteraciones):
        for individuo in individuos:
            if individuo.en_meta:
                continue

            individuo.iteraciones += 1

            pos_anterior = (individuo.y, individuo.x)

            movimiento = np.random.choice(len(movimientos), p=individuo.probs)

            dx, dy = movimientos[movimiento]
            new_x = max(0, min(grida_dim - 1, individuo.x + dx))
            new_y = max(0, min(grida_dim - 1, individuo.y + dy))

            if (new_y, new_x) in posiciones_ocupadas:
                continue

            individuo.x = new_x
            individuo.y = new_y

            if individuo.x == meta:
                individuo.en_meta = True
                individuos_en_meta.append(individuo)

            if pos_anterior in posiciones_ocupadas:
                posiciones_ocupadas.remove(pos_anterior)
            posiciones_ocupadas.add((individuo.y, individuo.x))

            grida[pos_anterior] = color_fondo
            grida[individuo.y, individuo.x] = individuo.color

        visualizar_grida(grida, gen, i)

    # Si un individuo llega a la meta, generamos más individuos y conservamos el que llegó a la meta
    
    if individuos_en_meta and len(individuos_en_meta) == 0:
        individuos = [ind for ind in individuos if not ind.en_meta]
    elif individuos_en_meta and len(individuos_en_meta) == 1:
        individuos = [ind for ind in individuos if not ind.en_meta]
    elif individuos_en_meta and len(individuos_en_meta) < 3:
        nuevos_individuos = generar_hijos(individuos_en_meta, num_individuos - 1, p=0.7)
        for nuevo_individuo in nuevos_individuos:
            posiciones_ocupadas.add((nuevo_individuo.y, nuevo_individuo.x))
        individuos = nuevos_individuos
        individuos += individuos_en_meta
    #individuos = [ind for ind in individuos if not ind.en_meta]

print("Individuos que llegaron a la meta:")
for individuo in individuos_en_meta:
    print(f"Color: {individuo.color}, Iteraciones: {individuo.iteraciones}, Genética: {individuo.probs}")
