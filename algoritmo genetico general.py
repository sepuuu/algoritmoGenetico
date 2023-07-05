import numpy as np
import matplotlib.pyplot as plt

plt.ion()

grida_dim = 20
grida = np.full((grida_dim, grida_dim, 3), 255)

individuos = []
movimientos = [(0, -1), (0, 1), (1, 0), (-1, 0), (1, -1), (-1, -1), (1, 1), (-1, 1), (0, 0)]
colores = [(128, 0, 0), (139, 0, 0), (165, 42, 42), (178, 34, 34), (220, 20, 60), (255, 0, 0), (255, 99, 71), (255, 165, 0), (255, 140, 0)]
num_individuos = 40
color_fondo = [255, 255, 255]
num_generaciones = 20
num_generacion = 1 #es para el conteo de las generaciones y que este conteo sea global
num_salto_generaciones = 5
# Para almacenar los datos requeridos durante las generaciones
individuos_en_meta_generaciones = []
pasos_promedio_generaciones = []
pasos_primer_en_meta_generaciones = []

class Individuo:
    def __init__(self, x, y, probs, color):
        self.x = x
        self.y = y
        self.probs = probs
        self.color = color
        self.en_meta = False
        self.pasos = 0

    def fitness(self):
        if self.pasos == 0:
            return 0
        else:
            return  self.pasos
        
def probabilidades(individuos):
    p = 0.5 # o el valor que desees
    probabilities = [p * ((1 - p) ** i) for i in range(len(individuos))]
    total = sum(probabilities)
    return [prob / total for prob in probabilities]

def seleccion_por_aptitud(individuos):
    individuos.sort(key=lambda ind: ind.fitness(), reverse=False)
    probabilities = probabilidades(individuos)
    individuo_seleccionado = np.random.choice(individuos, p=probabilities)
    return individuo_seleccionado


def cruzar_un_punto(ind1, ind2):
    punto_cruce = np.random.randint(len(ind1.probs))
    new_probs1 = np.concatenate((ind1.probs[:punto_cruce], ind2.probs[punto_cruce:]))
    new_probs2 = np.concatenate((ind2.probs[:punto_cruce], ind1.probs[punto_cruce:]))

    # Cambiamos las posiciones iniciales de los hijos
    x1 = np.random.randint(0, grida_dim//10)
    y1 = np.random.randint(0, grida_dim)
    x2 = np.random.randint(0, grida_dim//10)
    y2 = np.random.randint(0, grida_dim)

    hijos = [Individuo(x1, y1, new_probs1, ind1.color), Individuo(x2, y2, new_probs2, ind2.color)]
    return hijos

def mutar(individuo, tasa_mutacion=0.1):
    cambio = tasa_mutacion * np.random.randn(len(movimientos))
    individuo.probs = individuo.probs + cambio
    individuo.probs = np.clip(individuo.probs, 0, None) # limitar las probabilidades a no ser negativas
    individuo.probs = individuo.probs / np.sum(individuo.probs) # normalizar las probabilidades
    movimiento_predominante = np.argmax(individuo.probs)
    individuo.color = colores[movimiento_predominante]

seed = 1
np.random.seed(seed)
for _ in range(num_individuos):
    x = np.random.randint(0, grida_dim//10)
    y = np.random.randint(0, grida_dim)
    probs = np.random.rand(len(movimientos))
    probs = probs / np.sum(probs)
    movimiento_predominante = np.argmax(probs)
    individuo = Individuo(x, y, probs, colores[movimiento_predominante])
    individuos.append(individuo)
    grida[y, x] = individuo.color

meta = grida_dim - 1

fig, ax = plt.subplots()  # Creamos la figura y los ejes al inicio del programa

def visualizar_grida(grida,num_iteracion):
    ax.cla()  # Limpiamos los datos de la gráfica
    ax.imshow(grida, origin='lower')
    plt.grid(color='gray', linestyle='-', linewidth=0.2)
    plt.title(f"Generación: {num_generacion}, Iteracion: {num_iteracion}") 
    plt.draw()
    plt.pause(0.001)

visualizar_grida(grida,1)

individuos_en_meta = []
num_iteraciones = 50
fitness_promedio_generaciones = []

def ejecutar_generacion(individuos):
    grida = np.full((grida_dim, grida_dim, 3), 255)
    posiciones_ocupadas = set((ind.y, ind.x) for ind in individuos)
    individuos_en_meta = []
    primer_en_meta = None

    for ind in individuos:
        grida[ind.y, ind.x] = ind.color

    num_iteracion = 1
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
            individuo.pasos += 1

            if individuo.x == meta:
                individuos_en_meta.append(individuo)
                individuo.en_meta = True
                if primer_en_meta is None:
                    primer_en_meta = individuo

            if pos_anterior in posiciones_ocupadas:
                posiciones_ocupadas.remove(pos_anterior)
            posiciones_ocupadas.add((individuo.y, individuo.x))

            grida[pos_anterior] = color_fondo
            grida[individuo.y, individuo.x] = individuo.color
        num_iteracion = i + 1

        if(num_generacion%num_salto_generaciones == 0 or num_generacion==1):
            visualizar_grida(grida,num_iteracion)

        fig.canvas.draw()  # Dibuja la gráfica
        fig.canvas.flush_events()  # Actualiza la gráfica
        fitness_promedio = np.mean([ind.fitness() for ind in individuos])
        fitness_promedio_generaciones.append(fitness_promedio)

    print(f"Individuos que alcanzaron la meta: {len(individuos_en_meta)}")
    for ind in individuos_en_meta:
        print(f"Fitness de individuo: {ind.fitness()}")

    return individuos_en_meta, primer_en_meta


# Al inicio, creamos los individuos aleatoriamente
individuos = []
for _ in range(num_individuos):
    x = np.random.randint(0, grida_dim//10)
    y = np.random.randint(0, grida_dim)
    probs = np.random.rand(len(movimientos))
    probs = probs / np.sum(probs)
    movimiento_predominante = np.argmax(probs)
    individuo = Individuo(x, y, probs, colores[movimiento_predominante])
    individuos.append(individuo)
# Para almacenar los datos requeridos durante las generaciones
individuos_en_meta_generaciones = []
pasos_promedio_generaciones = []
pasos_primer_en_meta_generaciones = []


#bucle principal
for i in range(num_generaciones):
    num_generacion = i + 1 
    print(f"Ejecutando generación {i+1}")
    
    individuos_en_meta, primer_en_meta = ejecutar_generacion(individuos)

    if len(individuos_en_meta) < 2:
        print(f"Sólo {len(individuos_en_meta)} individuos alcanzaron la meta en la generación actual. No se repetirá la generación.")

    # Guardando la información requerida de esta generación
    individuos_en_meta_generaciones.append(len(individuos_en_meta))
    if len(individuos_en_meta) > 0:
        pasos_promedio_generaciones.append(np.mean([ind.pasos for ind in individuos_en_meta]))
        pasos_primer_en_meta_generaciones.append(primer_en_meta.pasos)
    else:
        pasos_promedio_generaciones.append(0)
        pasos_primer_en_meta_generaciones.append(0)

    print(f"Individuos que alcanzaron la meta: {len(individuos_en_meta)}")
    if len(individuos_en_meta) > 0:
        probabilidades1 = probabilidades(individuos_en_meta)
        for i, ind in enumerate(individuos_en_meta):
            print(f"Individuo {i+1}: Fitness: {ind.fitness()}, Pasos: {ind.pasos}, Probabilidad de ser seleccionado: {probabilidades1[i]}")

    if len(individuos_en_meta) > 0:
        nuevos_individuos = []
        if primer_en_meta is not None:
            x = np.random.randint(0, grida_dim//10)
            y = np.random.randint(0, grida_dim)
            nuevos_individuos.append(Individuo(x, y, primer_en_meta.probs.copy(), (0, 0, 255)))
        while len(nuevos_individuos) < num_individuos:
            padre1 = seleccion_por_aptitud(individuos_en_meta)
            padre2 = seleccion_por_aptitud(individuos_en_meta)
            hijos = cruzar_un_punto(padre1, padre2)
            for hijo in hijos:
                mutar(hijo)
                nuevos_individuos.append(hijo)
        individuos = nuevos_individuos
    else:
        print("Ningún individuo alcanzó la meta en la generación actual.")



plt.ioff()
plt.show() 

generaciones = list(range(1, num_generaciones + 1))

# Gráfico 1: cantidad de individuos que llegan con respecto a la generación actual
plt.figure()
plt.plot(generaciones, individuos_en_meta_generaciones)
plt.title('Cantidad de individuos que llegan por generación')
plt.xlabel('Generación')
plt.ylabel('Cantidad de individuos que llegan')
plt.show()

# Gráfico 2: pasos promedios de TODOS los individuos que llegan a la meta(borde derecho) con respecto a su generación(grafico de barras)
plt.figure()
plt.bar(generaciones, pasos_promedio_generaciones)
plt.title('Pasos promedio de individuos que llegan por generación')
plt.xlabel('Generación')
plt.ylabel('Pasos promedio')
plt.show()

# Gráfico 3: cantidad de pasos del primero de cada generación(haz un grafico de barras)
plt.figure()
plt.bar(generaciones, pasos_primer_en_meta_generaciones)
plt.title('Cantidad de pasos del primero en llegar por generación')
plt.xlabel('Generación')
plt.ylabel('Cantidad de pasos del primero')
plt.show()

plt.ioff()
plt.show() 

print(f"Individuos que alcanzaron la meta: {len(individuos_en_meta)}")
probabilidades1 = probabilidades(individuos_en_meta)

for i, ind in enumerate(individuos_en_meta):
    print(f"Individuo {i+1}: Fitness: {ind.fitness()}, Pasos: {ind.pasos}, Probabilidad de ser seleccionado: {probabilidades1[i]}")

