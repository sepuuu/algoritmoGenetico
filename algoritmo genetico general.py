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
num_generacion = 1 
num_salto_generaciones = 20 
individuos_en_meta_generaciones = []
pasos_promedio_generaciones = []
pasos_primer_en_meta_generaciones = []
seed = 1 #obtener resultados reproducibles y poder hacer un estudio
np.random.seed(seed)
meta = grida_dim - 1
fig, ax = plt.subplots()  # Creamos la figura y los ejes al inicio del programa


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
        
def probabilidades(individuos): #calculo de probabilidades para la seleccion de individuos con respecto a su fitness
    p = 0.5 # o el valor que desees
    probabilities = [p * ((1 - p) ** i) for i in range(len(individuos))]
    total = sum(probabilities)
    return [prob / total for prob in probabilities]

def seleccion_por_aptitud(individuos):# Realiza una selección de individuos por aptitud (fitness) basándose en sus probabilidades de selección.
    individuos.sort(key=lambda ind: ind.fitness(), reverse=False)
    probabilities = probabilidades(individuos)
    individuo_seleccionado = np.random.choice(individuos, p=probabilities)
    return individuo_seleccionado

def cruzar_un_punto(ind1, ind2):#Realiza el cruce de dos individuos en un punto aleatorio.
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

def mutar(individuo, tasa_mutacion=0.1): #Aplica una mutación a las probabilidades de movimiento de un individuo.
    cambio = tasa_mutacion * np.random.randn(len(movimientos)) #Genera un arreglo de cambios aleatorios utilizando la distribución normal (Gaussiana) con media cero y desviación estándar igual a tasa_mutacion. La longitud del arreglo de cambios es igual al número de movimientos posibles.
    individuo.probs = individuo.probs + cambio #actualizamos probabilidades de movimiento 
    individuo.probs = np.clip(individuo.probs, 0, None) # limitar las probabilidades a no ser negativas
    individuo.probs = individuo.probs / np.sum(individuo.probs) # normalizar las probabilidades
    movimiento_predominante = np.argmax(individuo.probs) 
    individuo.color = colores[movimiento_predominante] #actualizar el color despues de la mutacion

def visualizar_grida(grida,num_iteracion):
    ax.cla()  # Limpiamos los datos de la gráfica
    ax.imshow(grida, origin='lower')
    plt.grid(color='gray', linestyle='-', linewidth=0.2)
    plt.title(f"Generación: {num_generacion}, Iteracion: {num_iteracion}") 
    plt.draw()
    plt.pause(0.001)

visualizar_grida(grida, 1)

individuos_en_meta = []
num_iteraciones = 50
fitness_promedio_generaciones = []

def ejecutar_generacion(individuos): #simulamos una generacion de movimientos para los individuos
    grida = np.full((grida_dim, grida_dim, 3), 255)
    posiciones_ocupadas = set((ind.y, ind.x) for ind in individuos) #almacenamos las posiciones ocupadas en la generacion actual 
    individuos_en_meta = []
    primer_en_meta = None

    for ind in individuos: # se actualiza la grida con los colores de individuos actuales y en sus posiciones actuales
        grida[ind.y, ind.x] = ind.color

    num_iteracion = 1
    for i in range(num_iteraciones):
        
        for individuo in individuos: #iteramos sobre cada individuo en la lista individuos
            if individuo.en_meta: # si un individuo alcanza la meta que no siga moviendose 
                continue

            pos_anterior = (individuo.y, individuo.x) 

            movimiento = np.random.choice(len(movimientos), p=individuo.probs) #escogemos una posicion aleatoria 

            dx, dy = movimientos[movimiento]
            new_x = max(0, min(grida_dim - 1, individuo.x + dx)) #Se calcula la nueva posición del individuo considerando los límites de la cuadrícula. 
            new_y = max(0, min(grida_dim - 1, individuo.y + dy))

            if (new_y, new_x) in posiciones_ocupadas: #Si la nueva posición está ocupada por otro individuo, se continúa con el siguiente individuo. 
                continue

            individuo.x = new_x # se actualizan las coordenadas 
            individuo.y = new_y
            individuo.pasos += 1

            if individuo.x == meta: # se comprueba si llego a la meta y se pasa a True el atributo de la clase Individuo
                individuos_en_meta.append(individuo)
                individuo.en_meta = True
                if primer_en_meta is None:
                    primer_en_meta = individuo

            if pos_anterior in posiciones_ocupadas: # se elimina la posicion anterior del individuo para que no hayan colisiones
                posiciones_ocupadas.remove(pos_anterior)
            posiciones_ocupadas.add((individuo.y, individuo.x))

            grida[pos_anterior] = color_fondo # se actualizan los colores de la cuadricula segun la posicion nueva del individuo
            grida[individuo.y, individuo.x] = individuo.color
        num_iteracion = i + 1

        if(num_generacion%num_salto_generaciones == 0 or num_generacion==1):
            visualizar_grida(grida,num_iteracion) #mostrar iteracion en ploteo

        fig.canvas.draw()  # Dibuja la gráfica
        fig.canvas.flush_events()  # Actualiza la gráfica
 

    print(f"Individuos que alcanzaron la meta: {len(individuos_en_meta)}")
    for ind in individuos_en_meta:
        print(f"Fitness de individuo: {ind.fitness()}")

    return individuos_en_meta, primer_en_meta # retorna los individuos en la meta y el primero que llego a la meta para poder tomarlo hacia la siguiente gen

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
    grida[y, x] = individuo.color

# Para almacenar los datos requeridos durante las generaciones
individuos_en_meta_generaciones = []
pasos_promedio_generaciones = []
pasos_primer_en_meta_generaciones = []

#bucle principal
for i in range(num_generaciones):
    num_generacion = i + 1 
    print(f"Ejecutando generación {i+1}")
    
    while True: # Repite la generación hasta que al menos dos individuos alcancen la meta
        individuos_en_meta, primer_en_meta = ejecutar_generacion(individuos)

        if len(individuos_en_meta) >= 2:  # Si al menos dos individuos llegaron a la meta, sale del loop
            break
        else:
            print(f"Sólo {len(individuos_en_meta)} individuos alcanzaron la meta. Repitiendo la generación.")

    # Guardando la información requerida de esta generación
    individuos_en_meta_generaciones.append(len(individuos_en_meta)) #guardamos numero de individuos 
    
    print(f"Individuos que alcanzaron la meta: {len(individuos_en_meta)}")
    if len(individuos_en_meta) > 0:
        pasos_promedio_generaciones.append(np.mean([ind.pasos for ind in individuos_en_meta])) #calculamos el promedio de pasos para graficar 
        pasos_primer_en_meta_generaciones.append(primer_en_meta.pasos) #tambien guardamos para graficar
        probabilidades1 = probabilidades(individuos_en_meta) #Calcula las probabilidades de selección para los individuos que alcanzaron la meta en la generación actual.
        for i, ind in enumerate(individuos_en_meta): 
            print(f"Individuo {i+1}: Fitness: {ind.fitness()}, Pasos: {ind.pasos}, Probabilidad de ser seleccionado: {probabilidades1[i]}")
            # Imprime en pantalla la información del individuo, incluyendo su fitness, número de pasos y probabilidad de ser seleccionado
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
        pasos_promedio_generaciones.append(0) #graficar
        pasos_primer_en_meta_generaciones.append(0)

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