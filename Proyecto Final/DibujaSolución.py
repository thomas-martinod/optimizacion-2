import matplotlib.pyplot as plt
import numpy as np

def dibujaSolucion(individuo, datos = np.loadtxt("PuntosRio.csv", delimiter = ",")):
    x = datos[:,0]
    y = datos[:,1]


    nodos   = np.nonzero(individuo)[0]
    x_nodos = x[nodos]
    y_nodos = y[nodos]

    plt.plot(x,y)
    plt.plot(x_nodos,y_nodos,'-ok')

    plt.grid (True)

    plt.xlabel('x [m]', fontsize = 15)
    plt.ylabel('y [m]', fontsize = 15)
    plt.legend({'Río','Trazado óptimo'})

    # Misma escala en ambos ejes
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

I = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
dibujaSolucion(I, np.loadtxt("PuntosRio.csv", delimiter = ","))
