from math import sqrt

import numpy as np


# Essa é a classe mais básica de todas. Ela é um ponto
# e possui propriedades de um ponto, como coordenada x,
# coordenada y.
class Point():
    def __init__(self, x:float, y:float):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"({self.x}, {self.y})"
    
    # subtrarir p1 - p2 devolve um novo ponto (x2 - x1, y2 - y1)
    def __sub__(self, point):
        return Point(self.x - point.x, self.y - point.y)
    
    # somar p1 + p2 devolve um novo ponto (x1 + x2, y1 + y2)
    def __add__(self, point):
        return Point(self.x + point.x, self.y + point.y)

    # isso aqui serve para flexibilizar a sintaxe. Contextualmente,
    # não tem nada a ver com um ponto, então não se importe com isso
    def __iter__(self):
        yield self.x
        yield self.y
   
    # dividir p1 por um número n devolve um novo ponto (x1/n, y1/n)
    def __truediv__(self, number:int):
        return Point(self.x / number, self.y / number)
    
    # calcula a distância entre dois pontos (pitágoras das coordenadas)
    def dist(self, point) -> float:
        return sqrt( (self.x - point.x)**2 + (self.y - point.y)**2 )

# Essa classe representa um nó. Ela é um ponto (classe Point) com
# algumas características adicionais
class Node(Point):
    def __init__(self, x, y, value:int):
        super(Node, self).__init__(x, y)
        self.value = value # um ponto não tem a propriedade value, por exemplo
    
    # um ponto é representado apenas como tupla
    # um nó é representado como: "○ 1 (x, y)"
    def __repr__(self):
        return f"○ {self.value} ({self.x}, {self.y})"

# Essa classe representa um elemento. Ela possui 2 nós.
class Element():
    def __init__(self, node_1:Node, node_2:Node, value:int, E:float, A:float):
        self.value = value       # value de um elemento (elemento 1, elemento 2, ...)
        self.node_1 = node_1     # nó 1
        self.node_2 = node_2     # nó 2
        self.length = node_1.dist(node_2)  # comprimento do elemento
        self.cos, self.sin = self.slope()  # cos e sen do ângulo de inclinação
        
        self.m = np.array([[self.node_2.x - self.node_1.x], [self.node_2.y - self.node_1.y]])
        
        # self.S é a matriz de rigidez do elemento
        self.S = (E*A/self.length) * (self.m.dot(self.m.T)/(np.linalg.norm(self.m)**2))
        
    # inclinação
    def slope(self):
        cos, sin = (self.node_2 - self.node_1)/self.length
        return (cos, sin)
    
    def calc_strain(self, u1, v1, u2, v2):
        self.strain = np.array([ -self.cos, -self.sin, 
                                self.cos, self.sin ]).dot(np.array([[u1],[v1],[u2],[v2]]))/self.length
                                                                     
    @property
    def connectivity(self):
        connectivity_list = [0]*N_NODES
        connectivity_list[self.node_1.value - 1] = -1
        connectivity_list[self.node_2.value - 1] = 1
        return np.array( connectivity_list )

    def stiffness(self):
        c = np.array([self.connectivity])
        return np.kron(c.T.dot(c), self.S)
    
    def __repr__(self):
        return f"{self.node_1.value} ○--({self.value})--○ {self.node_2.value}"
