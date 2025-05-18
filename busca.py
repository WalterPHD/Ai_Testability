import networkx as nx
import matplotlib.pyplot as plt

def dfs_iterativo_niveis(grafo, inicio, objetivo):
    # Pilha para DFS: cada item é (nó_atual, caminho_percorrido)
    pilha = [(inicio, [inicio])]
    caminhos_por_nivel = []  # Guarda todos os caminhos completos de cada nível
    caminho_objetivo = []    # Guarda o caminho até o objetivo, se encontrado
    objetivo_encontrado = False
    nivel_objetivo = None    # Guarda o nível onde o objetivo foi encontrado

    while pilha:
        no_atual, caminho = pilha.pop()
        nivel = len(caminho) - 1  # O nível é o tamanho do caminho menos 1

        # Se já encontrou o objetivo e está no mesmo nível, ignora caminhos desse nível ou maiores
        if nivel_objetivo is not None and nivel >= nivel_objetivo:
            continue

        # Garante que existe uma lista para o nível atual
        if len(caminhos_por_nivel) <= nivel:
            caminhos_por_nivel.append([])

        # Adiciona o caminho completo ao nível se ainda não estiver lá
        if caminho not in caminhos_por_nivel[nivel]:
            caminhos_por_nivel[nivel].append(list(caminho))

        # Se encontrou o objetivo, salva o caminho e o nível, e impede novas expansões desse nível em diante
        if not objetivo_encontrado and no_atual == objetivo:
            objetivo_encontrado = True
            caminho_objetivo = caminho
            nivel_objetivo = nivel  # Marca o nível do objetivo

        # Adiciona vizinhos à pilha (ordem reversa para DFS da esquerda para direita)
        for vizinho in reversed(grafo.get(no_atual, [])):
            if vizinho not in caminho:  # Evita ciclos
                pilha.append((vizinho, caminho + [vizinho]))

    # Monta e imprime o vetor de nós únicos por nível (todos os visitados até o objetivo)
    print("\nVetor de nós únicos por nível:")
    for i, caminhos in enumerate(caminhos_por_nivel):
        nos_unicos = []
        for caminho in caminhos:
            for n in caminho:
                if n not in nos_unicos:
                    nos_unicos.append(n)
        print(f"Nível {i}: {nos_unicos}")
        # Para de mostrar níveis extras após encontrar o objetivo
        if nivel_objetivo is not None and i == nivel_objetivo:
            break

    # Monta a lista final de busca até o objetivo (primeiro caminho que chega ao objetivo)
    lista_final = []
    for caminhos in caminhos_por_nivel:
        for c in caminhos:
            for n in c:
                if n not in lista_final:
                    lista_final.append(n)
            if objetivo in c:
                break
        if objetivo in lista_final:
            break

    print("\nLista de busca até o objetivo:", lista_final)
    print("Caminho até o objetivo:", caminho_objetivo)
    return lista_final, caminho_objetivo, caminhos_por_nivel

def desenhar_grafo(grafo, caminho=None, lista_busca=None):
    # Cria o grafo direcionado usando networkx
    G = nx.DiGraph()
    for no, vizinhos in grafo.items():
        for vizinho in vizinhos:
            G.add_edge(no, vizinho)

    # Calcula posições hierárquicas para desenhar como árvore
    pos = hierarchy_pos(G, 'A')

    # Desenha o grafo completo
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=1500, font_size=16)

    # Destaca o caminho até o objetivo, se houver
    if caminho and len(caminho) > 1:
        edges_caminho = list(zip(caminho, caminho[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=edges_caminho, edge_color='red', width=3)
        nx.draw_networkx_nodes(G, pos, nodelist=caminho, node_color='orange')

    # Destaca os nós da lista de busca, se houver
    if lista_busca:
        nx.draw_networkx_nodes(G, pos, nodelist=lista_busca, node_color='yellow')

    plt.show()

def hierarchy_pos(G, root, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
    # Função auxiliar para desenhar o grafo em formato de árvore
    # Fonte: https://stackoverflow.com/questions/29586520/can-one-get-hierarchical-graphs-from-networkx-with-python-3
    if pos is None:
        pos = {root: (xcenter, vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)
    vizinhos = list(G.neighbors(root))
    if parent is not None and parent in vizinhos:
        vizinhos.remove(parent)
    if len(vizinhos) != 0:
        dx = width / len(vizinhos)
        nextx = xcenter - width / 2 - dx / 2
        for vizinho in vizinhos:
            nextx += dx
            pos = hierarchy_pos(G, vizinho, width=dx, vert_gap=vert_gap,
                                vert_loc=vert_loc - vert_gap, xcenter=nextx, pos=pos, parent=root)
    return pos

# Exemplo de uso:
# grafo = {
#     'A': ['B', 'C'],
#     'B': ['D', 'E'],
#     'C': ['F','G'],
#     'D': ['H', 'I'],
#     'E': ['J', 'K'],
#     'F': ['L', 'M'],
#     'G': ['N'],
# }
grafo = {
    'A': ['C', 'B'],
    'B': ['E', 'D'],
    'C': ['F', 'G'],
    'D': ['I', 'H'],
    'E': ['K', 'J'],
    'F': ['M', 'L'],
    'G': ['O', 'N'],
    'H': ['Q', 'P'],
    'I': ['S', 'R'],
    'J': ['U', 'T'],
    'K': ['W', 'V'],
    'L': ['Y', 'X'],
    'M': ['Z'],
    'N': [],
    'O': [],
    'P': [],
    'Q': [],
    'R': [],
    'S': [],
    'T': [],
    'U': [],
    'V': [],
    'W': [],
    'X': [],
    'Y': [],
    'Z': [],
}

# Executa a busca em profundidade iterativa até o objetivo 'W'
lista_busca, caminho_objetivo, caminhos_por_nivel = dfs_iterativo_niveis(grafo, 'A', 'J')

# Desenha o grafo completo
desenhar_grafo(grafo)

# Desenha o grafo mostrando a busca até o objetivo
desenhar_grafo(grafo, caminho=caminho_objetivo, lista_busca=lista_busca)

# 
# Sobre DFS iterativo:
# - Busca em profundidade (DFS) expande sempre o último nó inserido (pilha).
# - Útil para explorar caminhos até o fundo antes de voltar.
# - Referência: https://en.wikipedia.org/wiki/Depth-first_search
# - NetworkX: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.traversal.depth_first_search.dfs_edges.html

