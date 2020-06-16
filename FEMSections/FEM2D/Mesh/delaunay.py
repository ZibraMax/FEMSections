import triangle as tr

import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import leggauss as roots

from mpl_toolkits import mplot3d
from matplotlib import tri
import matplotlib as mpl
from matplotlib.lines import Line2D

class Geometria:
    def __init__(this, vertices):
        this.vertices = vertices
        this.cbe = []
        this.cbn = []
class Delaunay1V(Geometria):
    def __init__(this, vertices, params,plot=True):
        this.params = params
        super().__init__(vertices)
        this.seg = []
        for i in range(len(this.vertices)-1):
            this.seg.append([i,i+1])
        this.seg.append([i+1,0])
        this.original = dict(vertices=np.array(this.vertices),segments=np.array(this.seg))
        this.triangular = tr.triangulate(this.original,this.params)
        if plot:
            tr.compare(plt, this.original, this.triangular)
        count = 0
        for i in this.triangular['segments']:
            if count > 1:
                if np.sum(np.isin(np.array(this.cbe)[:,0], i[0]))<1:
                    this.cbe.append([i[0],0])
                if np.sum(np.isin(np.array(this.cbe)[:,0], i[1]))<1:
                    this.cbe.append([i[1],0])
            else:
                this.cbe.append([i[0],0])
            if np.sum(np.isin(np.array(this.cbe)[:,0], i[1]))<1:
                    this.cbe.append([i[1],0])
            count+=1
        this.diccionarios = this.triangular['triangles'].tolist()
        this.tipos = np.zeros([len(this.diccionarios)]).astype(str)
        this.tipos[:] = 'T1V'
        this.gdls = this.triangular['vertices'].tolist()
def Imesh(tf,tw,a,b,params,plot=True):
    corners = [[0,0],[a,0],[a,tf],[a/2+tw/2,tf],[a/2+tw/2,tf+b],[a,tf+b],[a,2*tf+b],[0,2*tf+b],[0,tf+b],[a/2-tw/2,tf+b],[a/2-tw/2,tf],[0,tf]]
    seg = []
    for i in range(len(corners)-1):
        seg.append([i,i+1])
    seg.append([i+1,0])
    A = dict(vertices=np.array(corners),segments=np.array(seg))
    B = tr.triangulate(A,params)
    if plot:
        tr.compare(plt, A, B)
    return B,corners

def _strdelaunay(constrained=True,delaunay=True,a=None,q=None):
    p = ''
    if constrained:
        p = 'p'
    if a == None:
        a = ''
    else:
        a = 'a'+format(a)
    D = ''
    if delaunay:
        D = 'D'
    if q == None:
        q=''
    else:
        if type(q) == int:
            if q > 35:
                raise "No sepuedecrearunatriangulacion conangulos menores a 35 grados"
        q = 'q'+format(q)
    return p+a+D+q+'i'
def generarGeometria(triang):
    cbe = []
    count = 0
    for i in triang['segments']:
        if count > 1:
            if np.sum(np.isin(np.array(cbe)[:,0], i[0]))<1:
                cbe.append([i[0],0])
            if np.sum(np.isin(np.array(cbe)[:,0], i[1]))<1:
                cbe.append([i[1],0])
        else:
            cbe.append([i[0],0])
        if np.sum(np.isin(np.array(cbe)[:,0], i[1]))<1:
                cbe.append([i[1],0])
        count+=1
    diccionarios = triang['triangles'].tolist()
    gdls = triang['vertices'].tolist()
    