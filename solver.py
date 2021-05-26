from math import sqrt

import numpy as np

from auxiliar import gauss_seidel, insert


class Point():
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"({self.x}, {self.y})"

    def __sub__(self, point):
        return Point(self.x - point.x, self.y - point.y)

    def __add__(self, point):
        return Point(self.x + point.x, self.y + point.y)

    def __iter__(self):
        yield self.x
        yield self.y

    def __truediv__(self, number: int):
        return Point(self.x / number, self.y / number)

    def dist(self, point) -> float:
        return sqrt((self.x - point.x)**2 + (self.y - point.y)**2)


class Node(Point):
    def __init__(self, x, y, value: int):
        super(Node, self).__init__(x, y)
        self.value = value

    def __repr__(self):
        return f"○ {self.value} ({self.x}, {self.y})"


class Element():
    def __init__(self, node_1: Node, node_2: Node, value: int, E: float, A: float):
        self.value = value
        self.node_1 = node_1
        self.node_2 = node_2
        self.E = E
        self.A = A
        self.length = node_1.dist(node_2)
        self.cos, self.sin = self.slope()

        self.m = np.array([[self.node_2.x - self.node_1.x],
                          [self.node_2.y - self.node_1.y]])

        self.S = (E*A/self.length) * \
            (self.m.dot(self.m.T)/(np.linalg.norm(self.m)**2))

    def slope(self):
        cos, sin = (self.node_2 - self.node_1)/self.length
        return (cos, sin)

    def calc_strain(self, u1, v1, u2, v2):
        self.strain = np.array([-self.cos, -self.sin,
                                self.cos, self.sin]).dot(np.array([[u1], [v1], [u2], [v2]]))/self.length

    def connectivity(self, N_NODES):
        self._N_NODES = N_NODES
        connectivity_list = [0]*N_NODES
        connectivity_list[self.node_1.value - 1] = -1
        connectivity_list[self.node_2.value - 1] = 1
        return np.array(connectivity_list)

    def stiffness(self):
        c = np.array([self.connectivity(self._N_NODES)])
        return np.kron(c.T.dot(c), self.S)

    def __repr__(self):
        return f"{self.node_1.value} ○--({self.value})--○ {self.node_2.value}"


class System():

    def __init__(self, E=None, A=None, debug=False):
        self._E = E
        self._A = A
        self._debug = debug
        self._nodes = []
        self._elements = []
        self._P = []
        self._loads = []
        self._constraints = []

    def add_node(self, x: float, y: float) -> None:
        new_node = Node(x, y, len(self._nodes) + 1)
        self._nodes.append(new_node)
        if self._debug:
            print(f"[SYS] created node: {new_node}")

    def add_constraint(self, node: int, direction: int):
        self._constraints = insert(self._constraints, node, direction)
        if self._debug:
            print(
                f"Constraint {'x' if direction == 1 else 'y'} added to node {int(node)}")

    def add_load(self, node, direction, value):
        self._loads.append([node, direction, value])
        if self._debug:
            print(
                f"Load of {value} N ({'x' if direction == 1 else 'y'}) added to node {node}")

    def define_element(self, n1: int, n2: int, E=None, A=None):
        if (E == None) and (self._E != None):
            E = self._E
        if (A == None) and (self._A != None):
            A = self._A
        n1 = self._nodes[n1 - 1]
        n2 = self._nodes[n2 - 1]
        new_element = Element(n1, n2, value=len(self._elements) + 1, E=E, A=A)
        self._elements.append(new_element)
        if self._debug:
            print(
                f"[SYS] created element: {new_element}; E = {E} Pa; A = {A} m²")

    def node(self, number: int) -> Node:
        return self._nodes[number - 1]

    @property
    def nodes(self):
        return np.array([[n.x, n.y] for n in self._nodes]).T

    def element(self, number: int) -> Element:
        return self._elements[number - 1]

    @property
    def N(self):
        return np.array([[n.x for n in self._nodes], [n.y for n in self._nodes]])

    @property
    def C(self):
        return np.array([el.connectivity(len(self._nodes))
                         for el in self._elements])

    @property
    def K(self):
        return sum([el.stiffness() for el in self._elements])

    @property
    def strain(self):
        return np.array([el.strain for el in self._elements])

    def solve(self, F, solver="gauss_seidel", tolerance=1e-7, max_iterations=1e3):
        P = F.copy()
        K = self.K.copy()

        for el in self._elements:
            el._N_NODES = len(self._nodes)

        deleted_positions = []

        for node, direction in self._constraints:
            P = np.delete(P, 2*(int(node) - 1) + int(direction)//2, 0)
            K = np.delete(K, 2*(int(node) - 1) + int(direction)//2, 0)
            K = np.delete(K, 2*(int(node) - 1) + int(direction)//2, 1)
            deleted_positions.append(2*(int(node) - 1) + int(direction)//2)

        if solver == "linalg":
            if self._debug:
                print(f"[SYS] Strategy used in solution: linalg.solve")
            solve = np.linalg.solve(K, P)
        else:
            if self._debug:
                print(f"[SYS] Strategy used in solution: Gauss-Seidel" +
                      f"| Max Iterations: {max_iterations} Tolerance: {tolerance}")
            solve = gauss_seidel(K, P, max_iterations, tolerance)

        U = np.zeros((2*len(self._nodes), 1))

        counter = 0
        for i in range(2*len(self._nodes)):
            if (i not in deleted_positions):
                U[i] = solve[counter]
                counter += 1

        R = self.K.dot(U)
        R[np.abs(R) < 1e-7] = 0

        for el in self._elements:
            values = U[[2*(el.node_1.value - 1), 2*(el.node_1.value - 1) + 1,
                        2*(el.node_2.value - 1), 2*(el.node_2.value - 1) + 1], 0]
            el.calc_strain(*values)

        return R, self.strain, U
