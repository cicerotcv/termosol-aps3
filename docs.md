# Código cometado

## Arquivo com as classes principais

```python
class Point():
    def __init__(self, x: float, y: float):
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
    def __truediv__(self, number: int):
        return Point(self.x / number, self.y / number)

    # calcula a distância entre dois pontos (pitágoras das coordenadas)
    def dist(self, point) -> float:
        return sqrt((self.x - point.x)**2 + (self.y - point.y)**2)

# Essa classe representa um nó. Ela é um ponto (classe Point) com
# algumas características adicionais
class Node(Point):
    def __init__(self, x, y, value: int):
        super(Node, self).__init__(x, y)
        self.value = value  # um ponto não tem a propriedade value, por exemplo

    # um ponto é representado apenas como tupla
    # um nó é representado como: "○ 1 (x, y)"
    def __repr__(self):
        return f"○ {self.value} ({self.x}, {self.y})"

# Essa classe representa um elemento. Ela possui 2 nós.
class Element():
    def __init__(self, node_1: Node, node_2: Node, value: int, E: float, A: float):
        # value de um elemento (elemento 1, elemento 2, ...)
        self.value = value
        self.node_1 = node_1     # nó 1
        self.node_2 = node_2     # nó 2
        self.E = E
        self.A = A
        self.length = node_1.dist(node_2)  # comprimento do elemento
        self.cos, self.sin = self.slope()  # cos e sen do ângulo de inclinação

        self.m = np.array([[self.node_2.x - self.node_1.x],
                          [self.node_2.y - self.node_1.y]])

        # self.S é a matriz de rigidez do elemento
        self.S = (E*A/self.length) * \
            (self.m.dot(self.m.T)/(np.linalg.norm(self.m)**2))

    # inclinação
    def slope(self):
        cos, sin = (self.node_2 - self.node_1)/self.length
        return (cos, sin)

    def calc_strain(self, u1, v1, u2, v2):
        self.strain = np.array([-self.cos, -self.sin,
                                self.cos, self.sin]).dot(np.array([[u1], [v1], [u2], [v2]]))/self.length

    def connectivity(self, N_NODES):
        self._N_NODES = N_NODES  # gambiarra
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

    # Esse método adiciona 1 nó de coordenadas (x, y) ao sistema
    def add_node(self, x: float, y: float) -> None:
        new_node = Node(x, y, len(self._nodes) + 1)
        self._nodes.append(new_node)
        if self._debug:
            print(f"[SYS] created node: {new_node}")

    # Esse método adiciona restrições aos nós (x: direction = 1; y: direction = 2)
    def add_constraint(self, node: int, direction: int):
        self._constraints = insert(self._constraints, node, direction)
        if self._debug:
            print(
                f"Constraint {'x' if direction == 1 else 'y'} added to node {int(node)}")

    # adiciona uma carga a um dos nós do sistema (x: direction = 1; y: direction = 2)
    def add_load(self, node, direction, value):
        self._loads.append([node, direction, value])
        if self._debug:
            print(
                f"Load of {value} N ({'x' if direction == 1 else 'y'}) added to node {node}")

    # Esse método adiciona 1 elemento definido entre os nós n1 e n2
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
            print(f"[SYS] created element: {new_element}; E = {E} Pa; A = {A} m²")

    # Esse método retorna um nó. Uso:
    # system.node(1) retorna o nó 1
    # Se 3 nós foram adicionados, eles são númerados como
    # 1, 2 e 3, então não se preocupe com índices a partir do 0
    def node(self, number: int) -> Node:
        return self._nodes[number - 1]

    @property
    def nodes(self):
        return np.array([[n.x, n.y] for n in self._nodes]).T

    # análogo ao anterior. system.element(1) retorna o elemento 1
    def element(self, number: int) -> Element:
        return self._elements[number - 1]

    # devolve a matriz dos nós
    @property
    def N(self):
        return np.array([[n.x for n in self._nodes], [n.y for n in self._nodes]])

    # devolve a matriz de conectividade
    @property
    def C(self):
        return np.array([el.connectivity(len(self._nodes))
                         for el in self._elements])

    # devolve a matriz
    @property
    def K(self):
        return sum([el.stiffness() for el in self._elements])

    @property
    def strain(self):
        return np.array([el.strain for el in self._elements])

    # usa linalg para resolver o sistema
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
```

## Arquivo de funções auxiliares

### Função que adiciona um elemento 1x2 em uma lista nx2

Essa função insere o elemento na lista mantendo a ordenação decrescente em
linha **E** coluna.

````python
# função que adiciona um um elemento 1x2 em uma lista nx2
# e mantém a ordenação descrescente
def insert(lista, node, direction):
    """```python
    insert( lista=[[4,2], [2,1]], node=3, direction=1 )
    # [[4,2], [3,1], [2,1]]
    ```"""
    # se a lista estiver vazia OU o último elemento tiver a primeira
    # posição maior que "node"
    if (len(lista) == 0) or (lista[-1][0] > node):
        lista.append([node, direction])
        return lista  # interrompe a execução
    # para cada elemento da lista
    for i, el in enumerate(lista):
        # Verifica se o elemento atual deve ser vir na frente de
        # [node, direction]. Para isso, verifica se a primeira
        # posição é menor que a do [n..,d..]. Nesse caso, o novo elemento
        # deve ser inserido logo em seguida. Se essa condição não
        # for verdadeira, entra na segunda sentença do "or" que
        # verifica se a primeira posição é
        if (el[0] < node) or (el[0] == node and el[1] < direction):
            # retorna uma lista do tipo
            # [todos até o indice i, novo elemento, todos após o índice i]
            return [*lista[:i], [node, direction], *lista[i:]]
    return lista
````

### Função que aplica o algoritmo de Gauss-Seidel

Essa função recebe como entradas a **matriz dos coeficientes** e a **matriz dos
coeficientes independentes**. A saída dessa função é a matriz das icógnitas.

Importante observar o formato das entradas:

```
A é uma matriz  (n, n)
b é uma matriz  (n, 1)
saída x é uma matrix (1, n)
```

Além disso, dois parâmetros adicionais `MAX_ITER` e `TOL` que representam o número
máximo de iterações que o algoritmo vai executar **OU** a variação relativa
entre uma iteração e outra mínima entre uma iteração e outra para que o
algoritmo continue, respectivamente. Se qualquer uma das condições for satisfeita,
ele é interrompido e o valor atual da matriz de icógnitas é devolvido.

```python
def gauss_seidel(A, b, MAX_ITER=100, TOL=1e-8):
    A = A.copy()
    b = b.copy()
    x = np.zeros((len(A), 1))
    D = np.diag(A)
    R = A - np.diagflat(D)

    x_copy = x.copy()

    for n_iter in range(MAX_ITER):
        for j in range(A.shape[0]):
            x[j, 0] = np.divide((b[j, 0] - np.sum(R[j, :]@x)), D[j])

        if (n_iter > 0):

            diff = b - A@x
            rel_diff = np.max(np.abs(np.divide(diff, b, where=b!=0)))

            if rel_diff < TOL:
                print(f"Tolerance reached in {n_iter} " +
                      f"iterations with relative diff {rel_diff}")
                return x

        x_copy = x.copy()

    print(f"Max number of iteration reached: {n_iter} with RD {rel_diff}")
    return x
```

Exemplo de uso:

```python
A1 = np.array([
        [3, 2],
        [2, 5]
    ])
b1 = np.array([[20, 39]])

print(gauss_seidel(A1, b1))
[[2. 7.]]

```
