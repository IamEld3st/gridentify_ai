from typing import List

# print a representation of a move to stdout
def show_move(move: List[int]) -> None: ...

class Gridentify:
    # getter for the board state, returns a list of 25 ints
    board: List[int] = ...
    # getter for the seed
    score: int = ...
    # initializes the game with the given seed
    @classmethod
    def __init__(cls, seed: int) -> None: ...
    # copies the whole game object (seed, board, score)
    def copy(self) -> Gridentify: ...
    # return a list of length 25 where each element has length 0-4
    def get_neighbours_of(self) -> List[List[int]]: ...
    # modify the board in-place, also updates the score
    def make_move(self, move: List[int]) -> None: ...
    # print a representation of the board to stdout
    def show_board(self) -> None: ...
    # return a list of valid moves, each move is a list of the used tiles in the order that they are used
    def valid_moves(self) -> List[List[int]]: ...