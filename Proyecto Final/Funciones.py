import numpy as np
import scipy as sp
import scipy.interpolate
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import matplotlib.pyplot as plt
import random

# Definimos el número de objetivos y las probabilidades
multi = False
pcx = 0.7
pmut = 0.3

# Generación de individuos
def crea_individuo(size=100):
    individuo = [0 for i in range(size)]
    x1 = random.randint(0, size - 1)
    x2 = random.randint(0, size - 1)
    if x1 > x2:
        x3 = x1
        x1 = x2
        x2 = x3
    elif x1 == x2:
        if x2 < size - 2:
            x2 = x2 + 2
        else:
            x1 = x1 - 2

    if x2 == (x1 + 1):
        if x2 < size - 2:
            x2 = x2 + 1
        else:
            x1 = x1 - 1

    individuo[x1:x2] = [1 for n in range(x2 - x1)]
    return individuo

# Operador de mutación
def mutFlipBitAs(individuo, indpb_01, indpb_10):
    for j, i in enumerate(individuo):
        if i == 1:
            if random.random() <= indpb_10:
                individuo[j] = 0

        if i == 0:
            if random.random() <= indpb_01:
                if random.random() <= indpb_01:
                    individuo[j] = 1

    return individuo

# Prestaciones de las plantas
def validaTrazado(individuo, datos):
    max_sop = 1.70
    max_exc = 1.20
    s = datos[:, 0]
    z = datos[:, 1]

    nodos = np.nonzero(individuo)[0]

    # Individuos de menos de 2 nodos no son válidos
    if len(nodos) < 2:
        return False

    s_nodos = s[nodos]
    z_nodos = z[nodos]
    trazado = sp.interpolate.interp1d(s_nodos, z_nodos)

    comprueba_sop = trazado(s[nodos[0]: nodos[-1]]) - z[nodos[0]: nodos[-1]]
    comprueba_exc = -1 * comprueba_sop

    # Comprobación de restricciones
    if all(comprueba_sop <= max_sop) == False:
        return False
    if all(comprueba_exc <= max_exc) == False:
        return False

    return True

# Prestaciones (potencia y coste) de la planta
def validaPlanta(individuo, datos):
    Pmin = 7e3
    Dtf = 32e-2
    CO = 25
    RHO = 1e3
    G = 9.8
    F = 2e-3
    DNOZ = 22e-3
    SNOZ = (np.pi * DNOZ ** 2) / 4
    REND = 0.9
    s = datos[:, 0]
    z = datos[:, 1]
    nodos = np.nonzero(individuo)[0]
    s_nodos = s[nodos]
    z_nodos = z[nodos]
    Hg = z[nodos[-1]] - z[nodos[0]]  # Diferencia de altura
    Ltf = np.sum(np.sqrt((s_nodos[1:] - s_nodos[0:-1]) ** 2 + (z_nodos[1:] - z_nodos[0:-1]) ** 2))
    Nc = sum(individuo)
    P = REND * (RHO / (2 * SNOZ ** 2)) * (Hg / (1 / (2 * G * SNOZ ** 2) + F * Ltf / (Dtf ** 5))) ** (3 / 2)
    C = 700 * (Ltf + CO * Nc) * Dtf ** 2
    cumplePotencia = False
    if P >= Pmin:
        cumplePotencia = True
    return P, C, cumplePotencia

# Función de fitness para el problema monoobjetivo
def fitness_function_single(individuo, datos):
    penaliza = 1e5
    if validaTrazado(individuo, datos) == False:
        return penaliza,
    else:
        P, C, cumplePotencia = validaPlanta(individuo, datos)
    if cumplePotencia == False:
        return penaliza,
    return C

# Problema MO
def fitness_function_multiobjetivo(individuo, datos):
    penaliza = 1e5
    if validaTrazado(individuo, datos) == False:
        return -penaliza, penaliza,
    P, C, cumplePotencia = validaPlanta(individuo, datos)
    if cumplePotencia == False:
        return -penaliza, penaliza
    return P, C

# Algoritmo mono
def unico_objetivo_ga(c, m):
    # Los parámetros de entrada son la probabilidad de cruce y mutación
    NGEN = 100
    MU = 3e3
    LAMBDA = 3e3
    CXPB = c
    MUTPB = m
    pop = toolbox.ini_poblacion(n=MU)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    logbook = tools.Logbook()
    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats=stats, halloffame=hof, verbose=True)
    return pop, hof, logbook


# Algoritmo MO
def multi_objetivo_ga(c, m):
    # Los parámetros de entrada son la probabilidad de cruce y mutación
    NGEN = 100
    MU = 2e3
    LAMBDA = 2e3
    CXPB = c
    MUTPB = m
    pop = toolbox.ini_poblacion(n=MU)
    hof = toolS.ParetoFront()

    algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, halloffame=hof)

    return pop, hof

if multi == False:
    # p1: creación del problema
    creator.create("Problema 1", base.Fitness, weights = (-1,))

    #p2: Generación de individuos