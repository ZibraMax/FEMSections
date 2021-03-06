from .integradorRectangular import *

class CuadrilateroL(integradorRectangular):
    def __init__(this,coords,gdl=None,gauss=4):
        if len(coords) == 4:
            this.psis = lambda z,n: np.array([[1/4*(1-z)*(1-n)],
                                              [1/4*(1+z)*(1-n)],
                                              [1/4*(1+z)*(1+n)],
                                              [1/4*(1-z)*(1+n)]])
            
            this.dzpsis = lambda z,n: np.array([[1/4*(n-1)],
                                                [-1/4*(n-1)],
                                                [1/4*(n+1)],
                                                [-1/4*(1+n)]])
            
            this.dnpsis = lambda z,n: np.array([[1/4*(z-1)],
                                                [-1/4*(z+1)],
                                                [1/4*(1+z)],
                                                [1/4*(1-z)]])
        else:
            raise Exception('Error: Se esta intentando crear un elemento que no tiene 4 coordenadas. '+
                           'Recuerde que se necesitan de 4 corrdenadas en sentido contrario de las manecillas del reloj '+
                           'sin repetir la primera coordenada')
        super().__init__(coords=coords,gdl=gdl,gauss=gauss)