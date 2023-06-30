import numpy as np
import matplotlib.pyplot as plt

plt.ion()

grida_dim = 20
grida = np.full((grida_dim, grida_dim, 3), 255)

individuos = []
# Movimientos posibles (N, S, E, O, NE, NW, SE, SW, Ninguno)
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

for _ in range(num_individuos):
    x = np.random.randint(0, grida_dim//4)
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

def visualizar_grida(grida):
    img.set_data(grida)
    plt.grid(color='gray', linestyle='-', linewidth=0.2)
    plt.draw()
    plt.pause(0.1)

visualizar_grida(grida)

individuos_en_meta = []
num_iteraciones = 1000

posiciones_ocupadas = set((ind.y, ind.x) for ind in individuos)

for i in range(num_iteraciones):
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

        individuo.x = new_x
        individuo.y = new_y

        if individuo.x == meta:
            individuos_en_meta.append(individuo)
            individuo.en_meta = True

        if pos_anterior in posiciones_ocupadas:
            posiciones_ocupadas.remove(pos_anterior)
        posiciones_ocupadas.add((individuo.y, individuo.x))

        grida[pos_anterior] = color_fondo
        grida[individuo.y, individuo.x] = individuo.color

    visualizar_grida(grida)

plt.ioff()
plt.show()


#falta implementar mutacion

#papa 1 = [a b c d e f g h i]
#papa 2 = [j k l m n o p q r]
#
#hijo 1 = [a b c m n o p q r]
#hijo 2 = [j k l d T f g h i]