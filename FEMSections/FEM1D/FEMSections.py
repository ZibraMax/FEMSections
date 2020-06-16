import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import leggauss as p_roots
from .Elemento import *
class FEMSections:
    """# Clase FEMSections
    ***
    Clase que define un problema de elementos finitos"""
    defaultConfig = {'numeroElementos': 40, 'ordenAproximacion': 1, 'puntosGauss': 3, 'h': 1 / 1000}
    """Configuracion por defecto para una instancia de la clase,
    esta se usara a menos de que se especifique otra configuracion.

    """
    x = lambda z, xa, he: ((xa + he) - xa) / 2 * z + (xa + (xa + he)) / 2
    """Funcion de transformacion de coordenadas naturales a globales

    """
    pesos = [0, 0, 0, 0, 0, 0]
    """Diccionario que contiene los pesos de la cuadratura de gauss
    para evaluarlos, cada indice corresponde a una lista 
    de los numeros de gauss a ser multiplicados, por ejemplo, si se
    escribe FEMSections.pesos[2] se obtienen los pesos correspondientes
    a integrar con 3 puntos en la cuadratura de gauss 

    """
    pesos[0] = [2]
    pesos[1] = [1, 1]
    pesos[2] = [5 / 9, 8 / 9, 5 / 9]
    pesos[3] = [(18 + np.sqrt(30)) / (36), (18 + np.sqrt(30)) / (36), (18 - np.sqrt(30)) / (36),
                (18 - np.sqrt(30)) / (36)]
    pesos[4] = [(322 + 13 * np.sqrt(70)) / (900), (322 + 13 * np.sqrt(70)) / (900), 128 / 225,
                (322 - 13 * np.sqrt(70)) / (900), (322 - 13 * np.sqrt(70)) / (900)]
    pesos[5] = [0.4679139346, 0.4679139346, 0.3607615730, 0.3607615730, 0.1713244924, 0.1713244924]

    puntos = [0, 0, 0, 0, 0, 0]
    """Diccionario que contiene los puntos de la cuadratura de gauss para
    evaluarlos, cada indice corresponde a una lista de los numeros de
    gauss a ser evaluados, por ejemplo, si se escribe FEMSections.puntos[2]
    se obtienen los puntos correspondientes a integrar con 3 puntos 
    en la cuadratura de gauss 
    

    """
    puntos[0] = [0]
    puntos[1] = [-np.sqrt(1 / 3), np.sqrt(1 / 3)]
    puntos[2] = [-np.sqrt(3 / 5), 0, np.sqrt(3 / 5)]
    puntos[3] = [-np.sqrt(3 / 7 - 2 / 7 * np.sqrt(6 / 5)), np.sqrt(3 / 7 - 2 / 7 * np.sqrt(6 / 5)),
                 -np.sqrt(3 / 7 + 2 / 7 * np.sqrt(6 / 5)), np.sqrt(3 / 7 + 2 / 7 * np.sqrt(6 / 5))]
    puntos[4] = [-1 / 3 * np.sqrt(5 - 2 * np.sqrt(10 / 7)), 1 / 3 * np.sqrt(5 - 2 * np.sqrt(10 / 7)), 0,
                 -1 / 3 * np.sqrt(5 + 2 * np.sqrt(10 / 7)), 1 / 3 * np.sqrt(5 + 2 * np.sqrt(10 / 7))]
    puntos[5] = [0.2386191861, -0.2386191861, 0.6612093865, -0.6612093865, 0.9324695142, -0.9324695142]

    def __init__(this, longitud, config=defaultConfig):
        """## Constructor
        ***

        Configura la sesion para resolver el problema con elementos finitos.

        Parameters
        ----------
        longitud: float
            Longitud del dominio del problema
        config: dict
            Configuracion del problema, por defecto es la variable FEMSections.defaultConfig


        Returns
        -------
        FEMSections
            Objeto de la clase FEMSections


        """
        this.config = config
        this.longitud = longitud
        this.establecerParametros(this.config)
        this.elementos = np.array([])
        this.mesh()
        this.tamaño = (this.ordenAproximacion) * (this.n) + 1

    def establecerParametros(this, config):
        """## FEMSections.establecerParametros(config)
        ***

        Establece losparametros de configuracion del problema, funcion privada.

        Parameters
        ----------
        config : dict
            Diccionario deconfiguracion


        """
        this.config = config
        this.n = config['numeroElementos']
        this.ordenAproximacion = config['ordenAproximacion']
        this.puntosGauss = config['puntosGauss']
        this.h = config['h']

    def definirCondicionesDeFrontera(this, cbe = [], cbn = [], cbc = []):
        """## FEMSections.definirCondicionesDeFrontera(cbe = [], cbn = [], cbc = [])
        ***

        Definicion de las condiciones de frontera en el dominio.
        
        Las condiciones de frontera deben darse en el siguiente formato:
        
        `cbe = [ [nodo,valorCondiciondeBorde] , [nodo,valorCondiciondeBorde] ]`
        
        `cbn = [ [nodo,valorCondiciondeBorde] , [nodo,valorCondiciondeBorde] ]`
        
        `cbc = [ [nodo,valorCondiciondeBorde] , [nodo,valorCondiciondeBorde] ]`
        
        
        Para facilitar introducir lascondiciones de frontera se puede usar el prefijo 0 para el primer nodo y el prefijo
        -1 para el ultimo nodo, por ejemplo, para un problema con condiciones de frontera escenciales en ambos extremos igual a 10.25:
        
        `cbe = [ [0,10.25] , [-1,10.25] ]`
        
        Para un problema con condicion de frontera escencial (=0 en este caso) en el nodo inicial y condicion de frontera natural en el nodo final ( :math:`-3.24\pi` en este caso) se tiene:
        
        `cbe = [ [0,0] ]`
        
        `cbn = [ [-1,-3.24*np.pi] ]`
        
        
        Si se tiene un problema con condiciones de frontera convectivas, por ejemplo:
        
        .. math::
            \\left(\\frac{d\\theta}{dx}+\\frac{\\beta}{k}\\theta)|_{x=L}=0
        donde
        
        .. math::
            \\theta = T-T_{\\infty}
        Esa condicion de frontera se puede expresar como:
        
        `cbn=[ [-1,Tinfty] ]`
        
        `cbc=[ [-1,beta/k] ]`

        Parameters
        ----------
        cbe : list
            Condiciones de borde escenciales
        cbn : list
            Condiciones de borde naturales
        cbc : list
            Coeficiente de condicion de borde convectiva
        

        """
        this.cbe = cbe
        this.cbn = cbn
        this.cbc = cbc

    def condicionesDeFronteraDesdeArchivo(this,ruta,separador=','):
        """## FEMSections.condicionesDeFronteraDesdeArchivo(ruta,separador=',')
        ***

        Convierte el archivo de coordenadas modificado a el formato interno de condiciones de borde.

        Parameters
        ----------
        ruta : str
            Ruta al archivo de coordenadas modificado.
        separador : list
            Separador de fila del archivo especificado. Por defecto es ',' ya que en los metodos que
            generan los archivos este es el separador decimal por defecto.
            

        """
        a = np.genfromtxt(ruta, delimiter=separador,skip_header=1)
        cbee = a[:,3]==1
        cbnn = a[:,4]==1
        cbcc = np.invert(a[:,5]==0)
        indicescbe = np.array(a[cbee][:,0]).astype(int)
        indicescbe = indicescbe.astype(int)
        cbe = np.array([indicescbe,a[cbee][:,2]]).T
        cbn = np.array([np.array(a[cbnn][:,0]).astype(int),a[cbnn][:,2]]).T
        cbc = np.array([np.array(a[cbcc][:,0]).astype(int),a[cbcc][:,5]]).T
        this.definirCondicionesDeFrontera(cbe.tolist(),cbn.tolist(),cbc.tolist())
    def mesh(this):
        """## FEMSections.mesh()
        ***

        Divide el dominio en elementos finitos basado en la configuracion actual. Se consideran nodos igualmente
        espaciados.
        Crea un arreglo que contiene cada uno de estos elementos.


        """
        if this.elementos.size == 0:
            he = this.longitud / this.n
            for i in range(0, this.n):
                gdl = np.arange(i * this.ordenAproximacion, (i + 1) * this.ordenAproximacion + 1, 1)
                elemento = Elemento(he, gdl, i * he, 1 / 1000)
                this.elementos = np.append(this.elementos, elemento)
        else:
            print('HEY PARCE YA HICISTE MESH, WTF?')

    def ecuacionesANivelDeElemento(this, a, c, f):
        """## FEMSections.ecuacionesANivelDeElemento(a,c,f)
        ***

        Crea las matrices de cada elemento, asi como el vector de fuerzas externas. Para ello se integra numericamente
        en el dominio del elemento segun las funciones indicadas por parametro. Las funciones pueden ser constantes o
        variables, por ejemplo:
        
        a = lambda x: 3*x+2
        
        a = lambda x: -20
        
        Para integrar se usa el numro de puntos de gauss que se especifican en las configuraciones del objeto FEMSections
        Se usan las funciones de forma que se indican segun el orden de aproximacion en las configuraciones.

        Parameters
        ----------
        a : function
            funcion a(x)
        c : function
            funcion c(x)
        f : function
            funcion f(x)


        """
        for i in this.elementos:
            i.Ke = FEMSections.ke(i, this.ordenAproximacion, this.puntosGauss, [a, c, f])
            i.Fe = FEMSections.fe(i, this.ordenAproximacion, this.puntosGauss, [a, c, f])

    def ensamblar(this):
        """## FEMSections.ensamblar()
        ***

        Ensambla una matriz general basado en los grados de libertad de todos los elementos.
        
        1. Crea el atributo K del objeto FEMSections que representa la matriz de coeficientes
        Esta matriz sera modificada al imponer las condiciones de frontera.
    
        2. Crea el atributo KORIGINAL que representa la matriz decoeficientes sin modificar
    
        3. Crea el vector de acciones externas a partir de los GDL de los elementos.
    
        4. Crea un vector de flujos Q de ceros
    
        5. Crea un vector temporal S para almacenar los valores previos a la solucion.
    
        6. Define el tamaño de cada uno de los vectores anteriores como: tamaño = ordenAproximacion*numeroDeElementos + 1
    
        7. Las matrices y vectores son arreglos de NumPy para facilitar sus operaciones vectoriales.

        8. Los elementos son ensamblados por sus extremos, no comparten nodos internos.
    

        """
        this.K = np.zeros([this.tamaño, this.tamaño])
        this.KORIGINAL = np.zeros([this.tamaño, this.tamaño])
        this.F = np.zeros([this.tamaño, 1])
        for i in this.elementos:
            this.K[np.ix_(i.gdl, i.gdl)] = this.K[np.ix_(i.gdl, i.gdl)] + i.Ke
            this.KORIGINAL[np.ix_(i.gdl, i.gdl)] = this.KORIGINAL[np.ix_(i.gdl, i.gdl)] + i.Ke
            this.F[np.ix_(i.gdl)] = this.F[np.ix_(i.gdl)] + i.Fe
        this.Q = np.zeros([this.tamaño, 1])
        this.S = np.zeros([this.tamaño, 1])

    def imponerCondicionesDeFrontera(this):
        """## FEMSections.imponerCondicionesDeFrontera()
        ***

        Impone las condiciones de frontera por el metodo exacto.
        Las modificaciones seran hechas en la matriz K y en el vector S, ya que estos seran los vectores posteriormetne operados.
        Los otros vectores no se modificaran para preservar futuras funcionalidades.


        """
        for i in this.cbn:
            this.Q[int(i[0])] = i[1]
        for i in this.cbc:
            this.Q[int(i[0])] = this.Q[int(i[0])] * i[1]
            this.K[int(i[0]), int(i[0])] = this.K[int(i[0]), int(i[0])] + i[1]
            this.KORIGINAL[int(i[0]), int(i[0])] = this.KORIGINAL[int(i[0]), int(i[0])] + i[1]
        for i in this.cbe:
            ui = np.zeros([this.tamaño, 1])
            ui[int(i[0])] = i[1]
            vv = np.dot(this.K, ui)
            this.S = this.S - vv
            this.K[int(i[0]), :] = 0
            this.K[:, int(i[0])] = 0
            this.K[int(i[0]), int(i[0])] = 1
        this.S = this.S + this.F + this.Q
        for i in this.cbe:
            this.S[int(i[0])] = i[1]
    
    def solucionarSistemaDeEcuaciones(this):
        """## FEMSections.solucionarSistemaDeEcuaciones()
        ***

        Soluciona el sistema de ecuaciones haciendo operaciones inversas de matrices con ayuda de la libreria NumPy
        Los resultados para la variable principal estaran en el atributo U del objeto FEMSections
        Los resultados de los flujos exteriores estaran en el atributo Qext del objeto FEMSections
        

        """
        this.U = np.dot(np.linalg.inv(this.K), this.S)
        this.Qext = np.dot(this.KORIGINAL, this.U) - this.F

    def postProceso(this):
        """## FEMSections.postProceso()
        ***

        Encuentra los valores para la variable secundaria en el borde de todos los elementos. Agrupa estos datos en
        el atributo Q del objeto FEMSections


        """
        this.Q = np.array([])
        for i in range(0, this.n):
            this.Q = np.append(this.Q, this.elementos[i].postProcesar(this.U)[0])
        this.elementos[-1].postProcesar(this.U)
        this.Q = np.append(this.Q, -this.elementos[-1].Qe[-1])

    def solucionar(this, a, c, f):
        """## FEMSections.solucionar()
        ***

        Agrupa los 6 pasos generales descritos anteriromente resolviendo por completo el problema con las configuraciones y funciones dadas

        Parameters
        ----------
        a : function
            Funcion a(x)

        c : function
            Funcion c(x)

        f : function
            Funcion f(x)
        

        """
        this.a = a
        this.c = c
        this.f = f
        if this.elementos.size == 0:
            this.mesh()
        this.ecuacionesANivelDeElemento(a, c, f)
        this.ensamblar()
        this.imponerCondicionesDeFrontera()
        this.solucionarSistemaDeEcuaciones()
        this.postProceso()
    def solucionInterpolada(this):
        """## FEMSections.solucionInterpolada()
        ***

        Interpola la solucion basandose en la definicion de las funciones de interpolacion.
        para ello usa el valor de la variable dependiente en cada uno de los nodos.

        Returns
        -------
        function
            Funcion evaluable en el dominio del problema. Esta funcion puede ser guaradada en
            cualquier variable. Por ejemplo, si se usa el meotodo como:
            `Ux = objetoFEMSEctions.solucionInterpolada()`
            Se puede posteriormente llamar a cualqier solucion en cualquier punto como:
            `Ux(0.03)`
            Esta funcion solamente arroja resultados si el punto donde se evalue se encuentre
            entre el dominio.
        

        """
        o = this.config['ordenAproximacion']
        he = this.elementos[0].he
        numero = this.elementos.size
        def Ux(x):
            if type(x)==float or type(x)==np.float64:
                for i in range(0,numero):
                    xa = this.elementos[i].xa
                    if x >= xa and x <= xa + he:
                        a = 0
                        for j in range(0,o+1):
                            a = a + this.elementos[i].Ue[j][0]*FEMSections.fi(x-xa, [j,o,he])
                        return a
            else:
                retorno = np.array([])
                for i in range(0,numero):
                    xa = this.elementos[i].xa
                    for l in range(0,x.size):
                        k = x[l]
                        if k > xa and k <= xa + he:
                            a = 0
                            for j in range(0,o+1):
                                a = a + this.elementos[i].Ue[j][0]*FEMSections.fi(k-xa, [j,o,he])
                            retorno = np.append(retorno, a)
                        elif k >= xa and k <= xa + he and xa==0:
                            a = 0
                            for j in range(0,o+1):
                                a = a + this.elementos[i].Ue[j][0]*FEMSections.fi(k-xa, [j,o,he])
                            retorno = np.append(retorno, a)
                return retorno
        return Ux
    def derivadaInterpolada(this):
        """## FEMSections.derivadaInterpolada()
        ***

        Interpola la derivada basandose en la definicion de las funciones de interpolacion.
        para ello usa el valor de la variable dependiente en cada uno de los nodos.

        Returns
        -------
        function
            Funcion evaluable en el dominio del problema. Esta funcion puede ser guaradada en
            cualquier variable. Por ejemplo, si se usa el meotodo como:
            `dUx = objetoFEMSEctions.derivadaInterpolada()`
            Se puede posteriormente llamar a cualqier solucion en cualquier punto como:
            `dUx(0.03)`
            Esta funcion solamente arroja resultados si el punto donde se evalue se encuentre
            entre el dominio.
        

        """
        o = this.config['ordenAproximacion']
        he = this.elementos[0].he
        numero = this.elementos.size
        def du(x):
            if type(x)==float or type(x)==np.float64:
                for i in range(0,numero):
                    xa = this.elementos[i].xa
                    if x >= xa and x <= xa + he:
                        a = 0
                        for j in range(0,o+1):
                            a = a + this.elementos[i].Ue[j][0]*FEMSections.dndxfi(x-xa, 1, [j,o,he])
                        return a
            else:
                retorno = np.array([])
                for i in range(0,numero):
                    xa = this.elementos[i].xa
                    for l in range(0,x.size):
                        k = x[l]
                        if k > xa and k <= xa + he:
                            a = 0
                            for j in range(0,o+1):
                                a = a + this.elementos[i].Ue[j][0]*FEMSections.dndxfi(k-xa, 1, [j,o,he])
                            retorno = np.append(retorno, a)
                        elif k >= xa and k <= xa + he and xa==0:
                            a = 0
                            for j in range(0,o+1):
                                a = a + this.elementos[i].Ue[j][0]*FEMSections.dndxfi(k-xa, 1, [j,o,he])
                            retorno = np.append(retorno, a)
                return retorno
        return du
    def estimacionError(this,tolerancia=1,clase='energia',avance=100,puntosGauss=1000,analitica=None,danalitica=None,tipo='he'):
        """## FEMSections.estimacionError(tolerancia=1,clase='energia',avance=100,puntosGauss=1000,analitica=None,danalitica=None,tipo='he')
        ***

        Estima el error indicado por parametro incrementando el numero de elementos en un avance indicado por parametro
        o comparando con la solucion analitica.

        Parameters
        ----------
        tolerancia : float
            factor C del termino chpe. Se estableció el nombre interno como tolerancia. En el caso de noindicarse se usa 1

        clase : str
            Clase del error a evaluar, este puede elegirse entre 'energia' y 'l2'. En el caso de no indicarse se usa 'energia'

        avance : int
            Numero de elementos para evaluar la solucion siguiente, En el caso de no indicarse se usa 100

        puntosGauss : int
            Numero de puntos de Gauss para integrar la solucion. Este factor es desicivo en la presicion del error, En el caso de no indicarse se usa 1000    

        analitica : function
            Funcion analitica del problema, en caso de que sea asignada el metodo no usará el avance para calcular el error.

        danalitica : function
            Derivada de la funcion analitica del problema, en caso de que sea asignada el metodo no usará el avance para calcular el error. En el caso de que el error sea clase 'l2' esta puede ser cualquier valor, ya que, no ser;a usada para el calculo.

        tipo : str
            Tipo de avance a realizar, se puede escoger entre 'he' y 'pe', la implementacion del error por 'pe' esta en desarrollo.

        Returns
        -------
        list
            Lista que contiene en su posicion 0 el valor del erorr y en su posicion 1 el valor para chpe
        

        """
        if tipo =='he':
            if not analitica == None and not danalitica ==None:
                u = analitica
                du = danalitica
            else:
                c = {'numeroElementos':this.config['numeroElementos'] + avance,'ordenAproximacion':this.config['ordenAproximacion'],'puntosGauss':this.config['puntosGauss'],'h':1/1000}
                temporal = FEMSections(longitud = this.longitud,config=c)
                temporal.definirCondicionesDeFrontera(this.cbe,this.cbn,this.cbc)
                temporal.solucionar(this.a,this.c,this.f)
                u = temporal.solucionInterpolada()
                du = temporal.derivadaInterpolada()
            uh = this.solucionInterpolada()
            duh = this.derivadaInterpolada()
            if clase == 'energia':
                fx = lambda z: (u(FEMSections.x(z,0,this.longitud))-uh(FEMSections.x(z,0,this.longitud)))**2+(du(FEMSections.x(z,0,this.longitud))*2/this.longitud-duh(FEMSections.x(z,0,this.longitud))*2/this.longitud)**2
            elif clase == 'l2':
                fx = lambda z: (u(FEMSections.x(z,0,this.longitud))-uh(FEMSections.x(z,0,this.longitud)))**2
            integral = FEMSections.integrarGauss(fx, puntosGauss, this.longitud)
            
            chep = tolerancia*this.elementos[0].he**this.config['ordenAproximacion']
            return np.sqrt(integral),chep

    def graficarSoluciones(this,nombreGrafica='Solucion para: ' + r'$-\frac{d}{dx}\left(a(x)\frac{du}{dx}\right)+c(x)u=f(x)$',
                           varaiblePrincipal='U', variableSecundaria='Q', unidadesVaraiblePrincipal='KN-m',
                           unidadesVaraibleSecundaria='KN', unidadesDistanciaElemento='m'):
        """## FEMSections,graficarSoluciones(nombreGrafica='Solucion para: ' + r'$-\frac{d}{dx}\left(a(x)\frac{du}{dx}\right)+c(x)u=f(x)$',varaiblePrincipal='U', variableSecundaria='Q', unidadesVaraiblePrincipal='KN-m',unidadesVaraibleSecundaria='KN', unidadesDistanciaElemento='m')
        ***

        Crea graficas para la solucion del problema. Requiere que el problema sea previamente resuleto (metodo solucionar)
        
        Se puede usar sintaxis LaTex pasando como cualquier parametro un string:  r'$Codigo-LaTex-Aqui$'
        Note la letra "r" antes de la definicion de las comillas del string

        Parameters
        ----------
        nombreGrafica : str
            Titulo de la grafica (Default value = 'Solucion para: ' + r'$-\frac{d}{dx}\left(a(x)\frac{du}{dx}\right)+c(x)u=f(x)$')

        varaiblePrincipal : str
            Nombre de la variable principal, por ejemplo U (Default value = 'U')

        variableSecundaria : str
            Nombre de la variable secundaria (Default value = 'Q')

        unidadesVaraiblePrincipal : str
            Unidades a mostrar de la varaible principal (Default value = 'KN-m')

        unidadesVaraibleSecundaria : str
            Unidades a mostrar de la variable secundaria (Default value = 'KN')

        unidadesDistanciaElemento : str
            Unidades del eje x del elemento (Default value = 'm')

            
        """
        fig, axes = plt.subplots(2, figsize=(15, 20))
        this.xs = np.array([])
        this.xsq = np.array([])
        for i in range(0, this.n):
            for j in range(0, this.ordenAproximacion):
                this.xs = np.append(this.xs,
                                    i * this.elementos[0].he + j * this.elementos[0].he / this.ordenAproximacion)
            this.xsq = np.append(this.xsq, i * this.elementos[0].he)
        this.xs = np.append(this.xs, this.n * this.elementos[0].he)
        this.xsq = np.append(this.xsq, this.n * this.elementos[0].he)
        fig.suptitle('[FEM] ' + nombreGrafica)
        axes[0].plot(this.xs, this.U)
        axes[1].plot(this.xsq, -this.Q)

        axes[0].set_title('Solución para ' + varaiblePrincipal)
        axes[0].set_xlabel('Posición [' + unidadesDistanciaElemento + ']')
        axes[0].set_ylabel(varaiblePrincipal + ' [' + unidadesVaraiblePrincipal + ']')
        axes[0].grid(b=True, which='major', color='#666666', linestyle='-')
        axes[1].set_title('Solución para ' + variableSecundaria)
        axes[1].set_xlabel('Posición [' + unidadesDistanciaElemento + ']')
        axes[1].set_ylabel(variableSecundaria + ' [' + unidadesVaraibleSecundaria + ']')
        axes[1].grid(b=True, which='major', color='#666666', linestyle='-')

    def guardarResultados(this,nombreArchivo='Resultados',tipo='csv',separador=','):
        """## FEMSections.guardarResultados(nombreArchivo='Resultados',tipo='csv',separador=',')
        ***

        Guarda los resultados del problema en un archivo de texto. El archivo sera guardado en el directorio actual.

        Parameters
        ----------
        nombreArchivo : str
            Nombre del archivo a ser guardado, en el caso de que no se especifique, se usa el nombre "Resultados"

        tipo : str
            Extension del archivo de guardado, se recomienda csv

        separador : str
            Separador de columnas del archivo. En el caso del que el tipo sea csv siempre se cambiara a ','
        

        """
        dudx = this.derivadaInterpolada()
        if tipo == 'csv' and not (separador == ','):
            separador = ','
            print('Se ha cambiado el separador decimal a "," porque el arcivo es .csv')
        a = 'Nudo'+separador+'X'+separador+'Varaible Dependiente'+separador+'Derivada Izquierda'+separador+'Derivada Derecha'+separador+'Varaible Secundaria Izquierda'+separador+'Variable Secundaria Derecha'+separador+'Flujo Externo'
        count = 0
        i = 0
        x = (this.longitud/(this.tamaño-1))*i
        a = a + '\n' + format(i) + separador + format(x) +separador+ format(this.U[i][0]) + separador + format(0) + separador + format(this.elementos[count].Qe[0][0]/this.a(x)) + separador + format(0) + separador + format(this.elementos[count].Qe[0][0]) + separador + format(this.Qext[i][0])
        for i in range(1,this.tamaño-1):
            count = count + (i%this.ordenAproximacion==0)*1
            x = (this.longitud/(this.tamaño-1))*i
            if not i%this.ordenAproximacion==0:
                a = a + '\n' + format(i) + separador + format(x) +separador+ format(this.U[i][0]) + separador + format(dudx(x)) + separador + format(dudx(x)) + separador + format(dudx(x)*this.a(x)) + separador + format(dudx(x)*this.a(x)) + separador + format(this.Qext[i][0])
            else:
                a = a + '\n' + format(i) + separador + format(x) +separador+ format(this.U[i][0]) + separador + format(this.elementos[count-1].Qe[-1][0]/this.a(x)) + separador + format(this.elementos[count].Qe[0][0]/this.a(x)) + separador + format(this.elementos[count-1].Qe[-1][0]) + separador + format(this.elementos[count].Qe[0][0]) + separador + format(this.Qext[i][0])
        i = this.tamaño-1
        count = 0
        x = (this.longitud/(this.tamaño-1))*i
        a = a + '\n' + format(i) + separador + format(x) +separador+ format(this.U[i][0]) + separador + format(this.elementos[count-1].Qe[-1][0]/this.a(x)) + separador + format(0) + separador + format(this.elementos[count-1].Qe[-1][0]) + separador + format(0) + separador + format(this.Qext[i][0])
        file = open(nombreArchivo+"."+tipo,"w+")
        file.write(a)
        file.close()
    def guardarArchivos(this,tipo='csv',separador=','):
        """## FEMSections.guardarArchivos(tipo='csv',separador=',')
        ***

        Guarda los archivos de definicion del problema. Entre ellos:
        ODE21D_1: Archivo de descripcion del problema.
        ODE21D_coord: Archivo que contiene condiciones de frontera
        ODE21D_conec: Conectividad entre elementos

        Parameters
        ----------
        tipo : str
            Extension del archivo de guardado, se recomienda csv

        separador : str
            Separador de columnas del archivo. En el caso del que el tipo sea csv siempre se cambiara a ','
        

        """
        this.guardarArchivoDefinicion(tipo,separador)
        this.guardarArchivoEnmallado(tipo,separador)
        this.guardarArchivoConectividad(tipo,separador)
    def guardarArchivoDefinicion(this,tipo='csv',separador=','):
        """## FEMSections.guardarArchivoDefinicion(tipo='csv',separador=',')
        ***

        Crea y guarda el archivo ODE21D_1

        Parameters
        ----------
        tipo : str
            Extension del archivo de guardado, se recomienda csv

        separador : str
            Separador de columnas del archivo. En el caso del que el tipo sea csv siempre se cambiara a ','


        """
        if tipo == 'csv' and not (separador == ','):
            separador = ','
            print('Se ha cambiado el separador decimal a "," porque el arcivo es .csv')
        a = 'longitud'+ separador + format(this.longitud)
        a = a + '\nnumeroElementos'+ separador + format(this.config['numeroElementos'])
        a = a + '\nordenAproximacion'+ separador + format(this.config['ordenAproximacion'])
        a = a + '\npuntosGauss'+ separador + format(this.config['puntosGauss'])

        file = open("ODE21D_1."+tipo,"w+")
        file.write(a)
        file.close()
    def guardarArchivoEnmallado(this,tipo='csv',separador=','):
        """## FEMSections.guardarArchivoEnmallado(tipo='csv',separador=',')
        ***

        Crea y guarda el archivo ODE21D_coord

        Parameters
        ----------
        tipo : str
            Extension del archivo de guardado, se recomienda csv

        separador : str
            Separador de columnas del archivo. En el caso del que el tipo sea csv siempre se cambiara a ','
        

        """
        if tipo == 'csv' and not (separador == ','):
            separador = ','
            print('Se ha cambiado el separador decimal a "," porque el arcivo es .csv')
        a = 'Nudo'+separador+'X'+separador+'VCB'+separador+'CBE'+separador+'CBN'+separador+'CRC'
        for i in range(0,this.tamaño-1):
            a = a + '\n' + format(i) + separador + format((this.longitud/(this.tamaño-1))*i) +separador+ '0'+separador+'0'+separador+'0'+separador+'0'
        a = a + '\n' + format(i+1) + separador + format((this.longitud/(this.tamaño-1))*(i+1)) +separador+ '0'+separador+'0'+separador+'0'+separador+'0'
        file = open("ODE21D_coord."+tipo,"w+")
        file.write(a)
        file.close()
    def guardarArchivoConectividad(this,tipo='csv',separador=','):
        """## FEMSections.guardarArchivoConectividad(tipo='csv',separador=',')
        ***

        Crea y guarda el archivo ODE21D_conec

        Parameters
        ----------
        tipo : str
            Extension del archivo de guardado, se recomienda csv

        separador : str
            Separador de columnas del archivo. En el caso del que el tipo sea csv siempre se cambiara a ','
        

        """
        if tipo == 'csv' and not (separador == ','):
            separador = ','
            print('Se ha cambiado el separador decimal a "," porque el arcivo es .csv')
        a = 'Elemento' +separador+ 'gdl'
        for i in range(0,this.elementos.size):
            a = a + '\n' + format(i) + separador + format(this.elementos[i].gdl)[1:-1]
        file = open("ODE21D_conec."+tipo,"w+")
        file.write(a)
        file.close()

    @staticmethod
    def ke(elemento, o, puntosGauss, funciones):
        """## FEMSections.ke(elemento, o, puntosGauss, funciones)
        ***

        Metodo estatico que define la matriz de un elemento que pase por parametro.

        Parameters
        ----------
        elemento : Elemento
            Elemento del quese quiere hallar la matriz de coeficientes
        o : int
            Orden de interpolacion de la matriz
        puntosGauss : int
            Numero de puntos de gauss para integrar, maximo 6
        funciones : list
            Arreglo de funciones conteniendo a, c, f respectivamente [a,c,f]

        Returns
        -------
        ndarray
            Matriz de coeficientes del elemento


        """
        a = funciones[0]
        c = funciones[1]
        f = funciones[2]
        he = elemento.he
        xa = elemento.xa
        h = elemento.h

        def kij(i, j):
            fx = lambda z: (a(FEMSections.x(z, xa, he)) * (FEMSections.dndxfiz(z, 1, [i, o, he], h)) * (
                FEMSections.dndxfiz(z, 1, [j, o, he], h)) * (2 / he) * (2 / he) + c(FEMSections.x(z, xa, he)) * FEMSections.fiz(z, [i, o,
                                                                                                                  he]) * FEMSections.fiz(
                z, [j, o, he]))
            return FEMSections.integrarGauss(fx, puntosGauss, he)

        k = np.zeros([o + 1, o + 1])
        for i in range(0, o + 1):
            for j in range(0, o + 1):
                k[i][j] = kij(i, j)
        return k

    @staticmethod
    def fe(elemento, o, puntosGauss, funciones):
        """## FEMSections.fe(elemento, o, puntosGauss, funciones)
        ***

        Metodo estatico que crea el vectoe de acciones externas de un elemento por parametro

        Parameters
        ----------
        elemento : Elemento
            Elemento del quese quiere hallar la matriz de coeficientes
        o : int
            Orden de interpolacion de la matriz
        puntosGauss : int
            Numero de puntos de gauss para integrar, maximo 6
        funciones : list
            Arreglo de funciones conteniendo a, c, f respectivamente [a,c,f]

        Returns
        -------
        ndarray
            Vector de fuerzas del elemento
        

        """
        a = funciones[0]
        c = funciones[1]
        f = funciones[2]
        he = elemento.he
        xa = elemento.xa
        h = elemento.h

        def fi(i):

            fx = lambda z: (f(FEMSections.x(z, xa, he)) * FEMSections.fiz(z, [i, o, he]))
            return FEMSections.integrarGauss(fx, puntosGauss, he)

        fvect = np.zeros([o + 1, 1])
        for i in range(0, o + 1):
            fvect[i] = fi(i)
        return fvect

    @staticmethod
    def fiz(z, param):
        """## FEMSections.fiz(z, param)
        ***

        Funcion de interpolacion en coordeandas naturales.

        Parameters
        ----------
        z : float
            Coordenada natural
        param : list
            lista que contenga [i,o,he] corresponde a i-esima funcion, orden de la funcion, tamaño del elemento consecutivamente

        Returns
        -------
        float
            Valor de la funcion evaluada en el punto z
        

        """
        # Reclasificacion de parametros para mejor lectura
        i = param[0]  # Numero de funcion de interpolación
        n = param[1]  # Orden de interpolacion
        he = param[2]  # Tamaño del elemento
        h = he / n  # Encuentra el tamaño de paso
        x = (1 + z) * (he / 2)
        a = 'Parametros Incorrectos'  # Valor por defecto en caso de un error en los parametros de entrada
        if i <= n:  # Evalua la condicion de la multiplicatoria
            a = 1  # Se inicializa la variable en 1 para no alterar el valor de la multiplicatoria
            for j in range(0, n + 1):  # Itera sobre cada uno de los ordenes de interpolacion. En Python la funcion
                # range(xo,xf) itera desde x0 hasta xf-1, por tal razon se suma 1 en el ciclo
                if not j == i:  # Evalua que no ocurra division por 0 siguiendo el metodo de lagrange
                    a = a * ((x - j * h) / (i * h - j * h))  # Opera la formula de la multiplicatoria
        return a  # Retorna el valor de la funcion
    @staticmethod
    def fi(x, param):
        """## FEMSections.fi(x,param)
        ***

        Funcion de interpolacion en coordeandas locales.

        Parameters
        ----------
        x : float
            Coordenada local
        param : list
            lista que contenga [i,o,he] corresponde a i-esima funcion, orden de la funcion, tamaño del elemento consecutivamente

        Returns
        -------
        float
            Valor de la funcion evaluada en el punto x
        

        """
        # Reclasificacion de parametros para mejor lectura
        i = param[0]  # Numero de funcion de interpolación
        n = param[1]  # Orden de interpolacion
        he = param[2]  # Tamaño del elemento
        h = he / n  # Encuentra el tamaño de paso
        a = 'Parametros Incorrectos'  # Valor por defecto en caso de un error en los parametros de entrada
        if i <= n:  # Evalua la condicion de la multiplicatoria
            a = 1  # Se inicializa la variable en 1 para no alterar el valor de la multiplicatoria
            for j in range(0, n + 1):  # Itera sobre cada uno de los ordenes de interpolacion. En Python la funcion
                # range(xo,xf) itera desde x0 hasta xf-1, por tal razon se suma 1 en el ciclo
                if not j == i:  # Evalua que no ocurra division por 0 siguiendo el metodo de lagrange
                    a = a * ((x - j * h) / (i * h - j * h))  # Opera la formula de la multiplicatoria
        return a  # Retorna el valor de la funcion
    @staticmethod
    def dndxfiz(x, n, param, h=1 / 100000):
        """## FEMSections.dndxfiz(x, n, param, h=1 / 100000)
        ***

        Derivada enesima de la funcion de aproximacion en coordenadas naturales usando diferencias finitas centradas. Dado que es comun el usar elementoslineales y elementos cuadraticos, estas derivadas se calcularon manualmente ys e tiene el valor exacto. Esto con el objetivo de evitar introducir errores al problema.

        Parameters
        ----------
        x : float
            punto de evaluacion en coordenadas naturales
        n : int
            orden de la derivada
        param : list
            lista que contenga [i,o,he] corresponde a i-esima funcion, orden de la funcion, tamaño del elemento consecutivamente
        h : float
            presicion de aproximacion, por defecto 1/100000 (Default value = 1 / 100000)

        Returns
        -------
        float
            Valor de laderivada enesima de la funcion de paroimacion
        

        """
        if (param[1] == 1  or param[1] == 2) and (n==1):
            if param[1]==1:
                if param[0]==0:
                    return -1/2
                else:
                    return 1/2
            elif param[1]==2:
                if param[0]==0:
                    return x - 1/2
                elif param[0]==1:
                    return -2*x
                else:
                    return x+1/2
        else:
            if n <= 0:  # Comprueba si el orden de diferenciacion es 0 para retornar el valor de la funcion
                return FEMSections.fiz(x, param)  # Retorna el valor de la funcion
            else:  # Recurre cambiando el orden de diferenciacion
                # Calcula diferencias finitas centradas de manera recursiva
                return (FEMSections.dndxfiz(x + h, n - 1, param, h) - FEMSections.dndxfiz(x - h, n - 1, param, h)) / (2 * h)
    @staticmethod
    def dndxfi(x, n, param, h=1 / 100000):
        """## FEMSections.dndxfi(x, n, param, h=1 / 100000)
        ***

        Derivada enesima de la funcion de aproximacion en coordenadas naturales usando diferencias finitas centradas. Dado que es comun el usar elementoslineales y elementos cuadraticos, estas derivadas se calcularon manualmente ys e tiene el valor exacto. Esto con el objetivo de evitar introducir errores al problema.

        Parameters
        ----------
        x : float
            punto de evaluacion en coordenadas naturales
        n : int
            orden de la derivada
        param : list
            lista que contenga [i,o,he] corresponde a i-esima funcion, orden de la funcion, tamaño del elemento consecutivamente
        h : float
            presicion de aproximacion, por defecto 1/100000 (Default value = 1 / 100000)

        Returns
        -------
        float
            Valor de laderivada enesima de la funcion de paroimacion


        """
        if (param[1] == 1  or param[1] == 2) and (n==1):
            he = param[2]
            if param[1]==1:
                if param[0]==0:
                    return -1/he
                else:
                    return 1/he
            elif param[1]==2:
                if param[0]==0:
                    return -3/he + 4*(x)/(he**2)
                elif param[0]==1:
                    return 4/he-8*x/(he**2)
                else:
                    return -1/he+4*x/he**2
        else:
            if n <= 0:  # Comprueba si el orden de diferenciacion es 0 para retornar el valor de la funcion
                return FEMSections.fi(x, param)  # Retorna el valor de la funcion
            else:  # Recurre cambiando el orden de diferenciacion
                # Calcula diferencias finitas centradas de manera recursiva
                return (FEMSections.dndxfi(x + h, n - 1, param, h) - FEMSections.dndxfi(x - h, n - 1, param, h)) / (2 * h)

    @staticmethod
    def desdeArchivo(ruta,separador=','):
        """## FEMSections.desdeArchivo(ruta,separador=',')
        ***

        Crea un objeto de la clase FEMSections a partir de un archivo de texto,

        Parameters
        ----------
        ruta : str
            Ruta al archivo input

        separador : str
            Separador de columnas del archivo

        Returns
        -------
        FEMSections
            Objeto que representa el problema con elementos finitos.

        """
        file = open(ruta,'r')
        rows = file.readlines()
        config = {}
        longitud = float(rows[0].split(separador)[1])
        for i in range(1,len(rows)):
            config[rows[i].split(separador)[0]] = int(rows[i].split(separador)[1])
        config['h']=1/1000000
        fem = FEMSections(longitud,config)        
        return fem
    @staticmethod
    def integrarGauss(fx, n, he,usarNumpy=True):
        """## FEMSections.integrarGauss(fx, n, he,usarNumpy=True)
        ***

        Integra la funcion fx (en coordenaaas naturales) con n puntos de gauss

        Parameters
        ----------
        fx : function
            Funcion a integrar, debe estar en coordenadas naturales
        n : int
            NUmero de puntos de gauss, maximo 6
        he : float
            Tamañao del elemento a integrar
        usarNumpy : bool
            Parametro que define la metodologia para evaluar los puntos y pesos de la cuadratura de Gauss.
            En el caso de que sea verdadero se usaran la funcion de la libreria NumPy esta no tiene limite de puntos.
            En el caso de que sea falso se usaran los diccionarios estblecidos en la deficion de esta clase. En este caso hay un limite de 6 puntos y pesos de gauss.
            Este paramtro por defecto es True y no se da la oportunidad de cambiarlo ya que se considera mucho mejor usar los de la lbreria de Numpy.
            En el caso de que se quieran usar los puntos de Gauss guardados en los arreglos este parametro debera ser cambiado desde este codigo.

        Returns
        -------
        float
            Valor de la integral

        """
        if usarNumpy:
            puntosYPesos = p_roots(n)
            puntos = puntosYPesos[0]
            pesos = puntosYPesos[1]
            return he / 2 * np.dot(np.array(pesos), fx(np.array(puntos)))
        else:
            return he / 2 * np.dot(np.array(FEMSections.pesos[n - 1]), fx(np.array(FEMSections.puntos[n - 1])))
