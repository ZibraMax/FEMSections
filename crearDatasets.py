import FEMSections as FEM
import FEMSections.FEM2D.Mesh as Mesh
from FEMSections.FEM2D.Utils import *
from FEMSections.FEM2D.Utils.error import *
import numpy as np
import matplotlib.pyplot as plt
import triangle as tr

def areasRefinado(modeloL,areas):
    a = []
    for e in modeloL.elementos:
        Ue = areas[np.ix_(e.gdl)]
        a.append(np.min(Ue))
        e.areaOptima = a[-1]
    return a
def extraerFila(modelo):
    m = []
    for e in modelo.elementos:
        coordsModelo = modelo.geometria.original['vertices'].flatten().tolist()
        cx = [np.average(e.coords[:,0])]
        cy = [np.average(e.coords[:,1])]
        coords = e.coords.flatten().tolist()
        cb = [np.any(np.isin(e.gdl,np.array(modelo.cbe)[:,0]))*1]
        y = [e.areaOptima]
        fila = coordsModelo+coords+cx+cy+cb+y
        m.append(fila)
    return m
import os
path = os.getcwd()

for i in range(176,180):
    try:
        ModeloL,ModeloH = generarPares(geom=1,da1=10,da2=80,bcb=[0,1,2])[0]
        errores,U,UG = L1error(ModeloH,ModeloL)
        areas = areaCorrection(ModeloL,errores,K=0.002, alpha=1,area0=0.5)
        areasR = areasRefinado(ModeloL,areas)
        ModeloL.geometria.triangular['triangle_max_area']=np.array(areasR).reshape([len(areasR),1])
        tnueva = tr.triangulate(ModeloL.geometria.triangular,'ra')
        tr.compare(plt,ModeloL.geometria.triangular,tnueva)
        plt.savefig('PROBLEMA'+format(i)+'.png')
        np.savetxt(path + "/PROBLEMA"+format(i)+".csv", np.array(extraerFila(ModeloL)), delimiter=",")
        plt.close('all')
    except:
        print('Fallo')