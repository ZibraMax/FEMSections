import numpy as np
class Elemento:
    """# Clase Elemento
    ***

    Define un elemento con porpia longitud, coordeandas, matrices de coeficientes, vectores de fuerzas, etc
    """

    def __init__(this, he, gdl, xa, h):
        """## Constructor
        ***

        Constructor de la clase Elemento, crea un nuevo elemento

        Parameters
        ----------
        he: float
            Longitud del elemento
        gdl: list
            Grados de libertad (nodos) en los que se encuentra el elemento
        xa: float
            xa: Coordenada incial del elemento
        h: float
            Presicion para calculos iterativos, por defecto es 1/1000


        Returns
        -------
        Elemento
            Objeto de la clase Elemento


        """
        this.he = he
        this.h = h
        this.gdl = gdl
        this.xa = xa

    def postProcesar(this, U):
        """## Elemento.postProcesar(U)
        ***

        Encuentra los flujos (variable secundaria) a nivel del elemento.

        Parameters
        ----------
        U : list
            Solucion del problema para todos los grados de libertad
        this :
            

        Returns
        -------
        list
            Flujos en el elemento


        """
        this.Ue = U[this.gdl]
        this.Qe = np.dot(this.Ke, this.Ue) - this.Fe
        return this.Qe