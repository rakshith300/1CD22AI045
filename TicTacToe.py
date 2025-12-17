# -*- coding: utf-8 -*-
"""
FAST Tic Tac Toe
Human vs AI
Input: 1 to 9
"""

import numpy as np
import random

BOARD_ROWS = 3
BOARD_COLS = 3


def num_to_pos(n):
    n -= 1
    return n // 3, n % 3


def pos_to_num(r, c):
    return r * 3 + c + 1


class State:
    def __init__(self):
        self.board = np.zeros((3, 3))
        self.playerSymbol = 1  # AI = 1, Human = -1
        self.isEnd = False

    def availablePositions(self):
        return [pos_to_num(i, j) for i in range(3) for j in range(3) if self.board[i, j] == 0]

    def updateState(self, action):
        r, c = num_to_pos(action)
        self.board[r, c] = self.playerSymbol
        self.playerSymbol = -1 if self.playerSymbol == 1 else 1

    def winner(self):
        for i in range(3):
            if abs(sum(self.board[i, :])) == 3:
                return np.sign(sum(self.board[i, :]))
            if abs(sum(self.board[:, i])) == 3:
                return np.sign(sum(self.board[:, i]))

        d1 = self.board[0, 0] + self.board[1, 1] + self.board[2, 2]
        d2 = self.board[0, 2] + self.board[1, 1] + self.board[2, 0]
        if abs(d1) == 3:
            return np.sign(d1)
        if abs(d2) == 3:
            return np.sign(d2)

        if len(self.availablePositions()) == 0:
            return 0
        return None

    def showBoard(self):
        print("-------------")
        for i in range(3):
            row = "| "
            for j in range(3):
                if self.board[i, j] == 1:
                    row += "X | "
                elif self.board[i, j] == -1:
                    row += "O | "
                else:
                    row += "  | "
            print(row)
            print("-------------")

    def showNumbers(self):
        print("""
1 | 2 | 3
---------
4 | 5 | 6
---------
7 | 8 | 9
""")


class AIPlayer:
    def chooseAction(self, state):
        # Try to win
        for p in state.availablePositions():
            r, c = num_to_pos(p)
            state.board[r, c] = 1
            if state.winner() == 1:
                state.board[r, c] = 0
                return p
            state.board[r, c] = 0

        # Block human
        for p in state.availablePositions():
            r, c = num_to_pos(p)
            state.board[r, c] = -1
            if state.winner() == -1:
                state.board[r, c] = 0
                return p
            state.board[r, c] = 0

        return random.choice(state.availablePositions())


class HumanPlayer:
    def chooseAction(self, positions):
        while True:
            try:
                move = int(input(f"Choose position {positions}: "))
                if move in positions:
                    return move
            except:
                pass
            print("Invalid move!")


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    print("TIC TAC TOE (FAST AI MODE)")

    while True:
        game = State()
        ai = AIPlayer()
        human = HumanPlayer()

        game.showNumbers()

        while True:
            # AI Move
            move = ai.chooseAction(game)
            game.updateState(move)
            game.showBoard()

            win = game.winner()
            if win is not None:
                print("AI Wins!" if win == 1 else "Draw!")
                break

            # Human Move
            move = human.chooseAction(game.availablePositions())
            game.updateState(move)
            game.showBoard()

            win = game.winner()
            if win is not None:
                print("You Win!" if win == -1 else "Draw!")
                break

        again = input("Play again? (y/n): ").lower()
        if again != 'y':
            print("Thanks for playing!")
            break
