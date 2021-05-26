import numpy as np


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


def gauss_seidel(A, b, MAX_ITER=100, TOL=1e-8):
    A = A.copy()
    b = b.copy()
    x = np.zeros((len(A), 1))
    D = np.diag(A)
    R = A - np.diagflat(D)

    done = False
    x_copy = x.copy()
    error = None

    n_iter = 0
    while n_iter < MAX_ITER:
        for j in range(A.shape[0]):
            x[j, 0] = np.divide((b[j, 0] - np.sum(R[j, :]@x)), D[j])

        if (n_iter > 0):

            error = b - A@x
            relative_error = np.max(np.abs(np.divide(error, b, where=b!=0)))

            if relative_error < TOL:
                print(f"Tolerance reached in {n_iter} " +
                      f"iterations with relative error {relative_error}")
                return x

        x_copy = x.copy()
        n_iter += 1

    print(f"Max number of iteration reached: {n_iter} with RE {relative_error}")
    return x


if __name__ == "__main__":
    # Testes
    # matrix 2x2 [ 2, 7, 5 ]
    A1 = np.array([
        [3, 2],
        [2, 5]
    ])
    b1 = np.array([[20, 39]])
    print("-"*8, "Saída 2x2", "-"*8)
    x1 = gauss_seidel(A1, b1, TOL=1e-12)
    print(x1.T)
    print("-"*8, "Gabarito", "-"*8)
    print((np.linalg.inv(A1)@b1.T).T)

    print('\n')

    # matrix 4x4 [ 1,  2, -1,  1]
    A2 = np.array([
        [10., -1., 2., 0.],
        [-1., 11., -1., 3.],
        [2., -1., 10., -1.],
        [0.0, 3., -1., 8.]
    ])
    b2 = np.array([[6., 25., -11., 15.]])
    print("-"*8, "Saída 4x4", "-"*8)
    x2 = gauss_seidel(A2, b2, TOL=1e-12)
    print(x2.T)
    print("-"*8, "Gabarito", "-"*8)
    print((np.linalg.inv(A2)@b2.T).T)

    print('\n')

    # [ 2, 7, 5 ]
    A3 = np.array([
        [3, 2, 1],
        [2, 5, 2],
        [3, 1, 2]
    ])
    b3 = np.array([[25, 49, 23]])
    print("-"*8, "Saída 3x3", "-"*8)
    x3 = gauss_seidel(A3, b3, TOL=1e-12)
    print(x3.T)
    print("-"*8, "Gabarito", "-"*8)
    print((np.linalg.inv(A3)@b3.T).T)
