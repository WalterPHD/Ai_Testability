def is_safe(board, row, col):
    # Verifica se há uma rainha na mesma linha à esquerda
    for i in range(col):
        if board[row][i] == 1:
            return False

    # Verifica a diagonal superior esquerda
    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False

    # Verifica a diagonal inferior esquerda
    for i, j in zip(range(row, len(board)), range(col, -1, -1)):
        if board[i][j] == 1:
            return False

    # Se não encontrou conflitos, é seguro colocar a rainha
    return True

def solve_n_queens_iterative(n):
    solutions = []  # Lista para armazenar todas as soluções encontradas
    stack = []      # Pilha para simular a recursão do DFS
    # Cria um tabuleiro vazio (n x n) preenchido com zeros
    board = [[0 for _ in range(n)] for _ in range(n)]
    # Adiciona o estado inicial à pilha: (coluna atual, tabuleiro, posições das rainhas)
    stack.append((0, board, []))

    # Enquanto houver estados na pilha, continue explorando
    while stack:
        # Remove o topo da pilha (DFS)
        col, curr_board, queens = stack.pop()

        # Se já colocou rainhas em todas as colunas, encontrou uma solução
        if col == n:
            # Converte o tabuleiro para uma representação visual (strings)
            solution = [''.join('Q' if cell == 1 else '.' for cell in row) for row in curr_board]
            solutions.append(solution)
            # break
            continue  # Volta para explorar outros caminhos

        # Tenta colocar uma rainha em cada linha da coluna atual (de baixo para cima)
        for row in range(n-1, -1, -1):  # Ordem invertida para DFS
            if is_safe(curr_board, row, col):
                # Cria uma cópia do tabuleiro para o novo estado
                new_board = [r[:] for r in curr_board]
                # Coloca a rainha na posição (row, col)
                new_board[row][col] = 1
                # Adiciona o novo estado à pilha para explorar a próxima coluna
                stack.append((col + 1, new_board, queens + [(row, col)]))

    # Retorna todas as soluções encontradas
    return solutions

import matplotlib.pyplot as plt

def plot_chessboard(solution):
    n = len(solution)
    fig, ax = plt.subplots(figsize=(6, 6))
    # Desenha o tabuleiro
    for i in range(n):
        for j in range(n):
            color = 'cornsilk' if (i + j) % 2 == 0 else 'saddlebrown'
            rect = plt.Rectangle([j, n - 1 - i], 1, 1, facecolor=color)
            ax.add_patch(rect)
            if solution[i][j] == 'Q':
                ax.text(j + 0.5, n - 1 - i + 0.5, '♛', fontsize=32, ha='center', va='center', color='black')
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    plt.title("Solução das 8 Rainhas")
    plt.show()

import random

if __name__ == "__main__":
    n = 8  # Número de rainhas e tamanho do tabuleiro
    solutions = solve_n_queens_iterative(n)
    print(f"Total solutions for {n}-Queens: {len(solutions)}")
    # Mostra uma solução aleatória graficamente
    if solutions:
        idx = random.randint(0, len(solutions) - 1)
        print(f"Mostrando solução aleatória número {idx + 1}")
        plot_chessboard(solutions[idx])