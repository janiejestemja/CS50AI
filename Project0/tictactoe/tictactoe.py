"""
Tic Tac Toe Player
"""

import math

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    count = 0
    for row in board:
        for field in row:
            if field != EMPTY:
                count += 1
    if count % 2 != 0:
        return O
    else:
        return X


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    aset = set()
    for i, row in enumerate(board):
        for j, field in enumerate(row):
            if field == EMPTY:
                aset.add((i, j))
    
    return aset


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    result = []
    for i, row in enumerate(board):
        new_row = []
        for j, field in enumerate(row):
            if action == (i, j)  and field != EMPTY:
                raise  ValueError
            if action != (i, j):
                new_row.append(field)
            else:
                new_row.append(player(board))
    
        result.append(new_row)
    return result


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # diagonals
    if board[1][1] != EMPTY:
        if board[0][0] == board[1][1] and board[1][1] == board[2][2]:
            return board[1][1]
        if board[0][2] == board[1][1] and board[1][1] == board[2][0]:
            return board[1][1]

    # columns and rows
    for i in range(3):
        if board[1][i] != EMPTY:
            if board[0][i] == board [1][i] and board[1][i] == board[2][i]:
                return board[1][i]
    for i in range(3):
        if board[i][1] != EMPTY:
            if board[i][0] == board [i][1] and board[i][1] == board[i][2]:
                return board[i][1]

    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board):
        return True

    for row in board:
        for field in row:
            if field == EMPTY:
                return False 
    return True


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == X:
        return 1
    elif winner(board) == O:
        return -1
    else:
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    alpha = -math.inf
    betha = math.inf

    if player(board) == X:
        v = -math.inf 
        for action in actions(board):
            m = min_val(result(board, action), alpha, betha)
            if m > v:
                v = m
                move = action 
    else:
        v = math.inf 
        for action in actions(board):
            m = max_val(result(board, action), alpha, betha)
            if m < v:
                v = m
                move = action
    return move 

def max_val(board, alpha, betha):
    if terminal(board):
        return utility(board)
    v = -math.inf
    for action in actions(board):
        v = max(v, min_val(result(board, action), alpha, betha))
        alpha = max(alpha, v)
        if v > betha:
            break
    return v

def min_val(board, alpha, betha):
    if terminal(board):
        return utility(board)
    v = math.inf 
    for action in actions(board):
        v = min(v, max_val(result(board, action), alpha, betha))
        betha = max(betha, v)
        if v < alpha:
            break
    return v
