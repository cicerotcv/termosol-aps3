

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
        lista.append([node,direction])
        return lista # interrompe a execução
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
