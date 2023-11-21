import matplotlib.pyplot as plt
import numpy as np

def dibujaRio(datos=np.loadtxt("rio2.csv",
     delimiter = ",")):

    x = datos[:,0]
    y = datos[:,1]

    plt.plot(x, y, 'b.')
    plt.grid (True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Perfil discreto del río')
    plt.xlabel('x [m]', fontsize = 15)
    plt.ylabel('y [m]', fontsize = 15)
    plt.show()

dibujaRio()