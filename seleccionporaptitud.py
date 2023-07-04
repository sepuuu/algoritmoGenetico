import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

grida_dim = 20
num_individuos = 50
num_iteraciones = 50  # Reducimos el número de iteraciones para la animación
num_generaciones = 5  # Reducimos el número de generaciones para la animación
colores = [(128, 0, 0), (139, 0, 0), (165, 42, 42), (178, 34, 34), (220, 20, 60), (255, 0, 0), (255, 99, 71), (255, 165, 0), (255, 140, 0)]
color_fondo = [255, 255, 255]
movimientos = [(0, -1), (0, 1), (1, 0), (-1, 0), (1, -1), (-1, -1), (1, 1), (-1, 1), (0, 0)]
seed = 11
np.random.seed(seed)
individuos = []
gridas = []
np.random.seed(seed)
class Individuo:
    def __init__(self, x, y, probs, color):
        self.x = x
        self.y = y
        self.probs = probs / np.sum(probs)  # Normalización de las probabilidades al crear el individuo
        self.color = color
        self.pasos = 0
        self.en_meta = False
        self.max_movimientos = 1000  # Agregar un límite al número de movimientos

    def mover(self, dx, dy):
        self.x = max(0, min(grida_dim - 1, self.x + dx))
        self.y = max(0, min(grida_dim - 1, self.y + dy))
        self.pasos += 1
        if self.pasos >= self.max_movimientos:  # Comprobar si el individuo ha alcanzado el límite de movimientos
            self.en_meta = True

def seleccion_por_aptitud(individuos_en_meta):
    num_padres = len(individuos_en_meta)
    fitness = np.array([ind.pasos for ind in individuos_en_meta])
    probs = 1 / fitness
    probs = probs / np.sum(probs)
    return np.random.choice(individuos_en_meta, size=2, p=probs)

def cruce_por_un_punto(padres):
    punto_de_cruce = np.random.randint(0, len(padres[0].probs))
    color_predominante = np.argmax(padres[0].probs)
    probs_hijo = np.concatenate([padres[0].probs[:punto_de_cruce], padres[1].probs[punto_de_cruce:]])
    hijo = Individuo(0, np.random.randint(0, grida_dim), probs_hijo / np.sum(probs_hijo), colores[color_predominante])  # Normalización de las probabilidades después del cruce
    return hijo

def una_generacion():
    global grida
    grida = np.full((grida_dim, grida_dim, 3), 255)
    global individuos
    individuos = []
    meta = grida_dim - 1
    individuos_en_meta = []
    posiciones_ocupadas = set()

    for _ in range(num_individuos):
        x = np.random.randint(0, grida_dim//4)
        y = np.random.randint(0, grida_dim)
        probs = np.random.rand(len(movimientos))
        probs = probs / np.sum(probs)
        movimiento_predominante = np.argmax(probs)
        individuo = Individuo(x, y, probs, colores[movimiento_predominante])
        individuos.append(individuo)
        grida[y, x] = individuo.color
        posiciones_ocupadas.add((y, x))
    for _ in range(num_iteraciones):
        for individuo in individuos:
            if individuo.en_meta:
                continue

            pos_anterior = (individuo.y, individuo.x)

            movimiento = np.random.choice(len(movimientos), p=individuo.probs)

            dx, dy = movimientos[movimiento]
            new_x = max(0, min(grida_dim - 1, individuo.x + dx))
            new_y = max(0, min(grida_dim - 1, individuo.y + dy))

            if (new_y, new_x) in posiciones_ocupadas:
                continue

            individuo.mover(dx, dy)

            if individuo.x == meta:
                individuos_en_meta.append(individuo)
                individuo.en_meta = True

            if pos_anterior in posiciones_ocupadas:
                posiciones_ocupadas.remove(pos_anterior)
            posiciones_ocupadas.add((individuo.y, individuo.x))

            grida[pos_anterior] = color_fondo
            grida[individuo.y, individuo.x] = individuo.color
            gridas.append(grida)
    return individuos_en_meta

def update(i):
    global grida
    grida = gridas[i]
    plt.imshow(grida)
    plt.title(f'Generación {i+1}')

historial_de_aptitud = []
for _ in range(num_generaciones):
    individuos_en_meta = una_generacion()
    nueva_generacion = []
    if individuos_en_meta:
        fitness_promedio = np.mean([ind.pasos for ind in individuos_en_meta])
        historial_de_aptitud.append(fitness_promedio)
        for _ in range(num_individuos - 1):
            padres = seleccion_por_aptitud(individuos_en_meta)
            hijo = cruce_por_un_punto(padres)
            nueva_generacion.append(hijo)
        nueva_generacion.append(min(individuos_en_meta, key=lambda ind: ind.pasos))
    else:
        # Si ningún individuo llega a la meta, rellenamos la generación con individuos aleatorios
        for _ in range(num_individuos):
            x = np.random.randint(0, grida_dim//4)
            y = np.random.randint(0, grida_dim)
            probs = np.random.rand(len(movimientos))
            probs = probs / np.sum(probs)
            movimiento_predominante = np.argmax(probs)
            individuo = Individuo(x, y, probs, colores[movimiento_predominante])
            nueva_generacion.append(individuo)
    individuos = nueva_generacion

fig = plt.figure()

ani = animation.FuncAnimation(fig, update, frames=num_generaciones, repeat=True)

plt.show()
