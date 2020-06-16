from .Elemento import *
class integradorTriangular(Elemento):
    def __init__(this,coords,gdl=None,gauss=7):
        super().__init__(coords=coords,gdl=gdl,gauss=7)
        DENSIDAD = 3
        trimesh = meshDeFigura([[0,0],[1,0],[0,1]],n=DENSIDAD,dev=False)
        this._dominioNaturalZ = trimesh.x
        this._dominioNaturalN = trimesh.y
    def intGauss2D(this,n,f):
        A0 = 1/3
        A1 = 0.059715871789770
        A2 = 0.797426985353087
        B1 = 0.470142064105115
        B2 = 0.101286507323456
        W0 = 0.1125
        W1 = 0.066197076394253
        W2 = 0.062969590272413
        X = [A0,A1,B1,B1,B2,B2,A2]
        Y = [A0,B1,A1,B1,A2,B2,B2]
        W = [W0,W1,W1,W1,W2,W2,W2]
        INT = 0
        for i in range(len(X)):
            INT += f(X[i],Y[i])*W[i]
        return INT
def meshDeFigura(corners,n=2,dev=False):
    corners = np.array(corners)
    triangle = tri.Triangulation(corners[:, 0], corners[:, 1])
    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=n)
    if dev:
        return trimesh,triangle
    else:
        return trimesh