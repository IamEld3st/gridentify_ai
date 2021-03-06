from gridentify import *
import numpy as np
import time

good_values = set([1,2,3,6,12,24,48,96,192,384,768,1536,3072,6144,12288,24578,49152])
good_move_lens = set([2,3,4,6,8,12,24])

# 39k
orig_weights = np.array([
        [ 128, 256, 512,1024,2048],
        [  64,  32,  16,   8,   4],
        [   2,   1,   0,   1,   2],
        [   4,   8,  16,  32,  64],
        [2048,1024, 512, 256, 128]
    ])

# 33k
one_corner_snail = np.array([
        [ 4, 3, 2, 1, 0],
        [ 5,12,13,14,15],
        [ 6,11,18,17,16],
        [ 7,10,19,22,23],
        [ 8, 9,20,21,24]
    ])
one_corner_snail = 2 ** one_corner_snail
one_corner_snail = one_corner_snail / 100

# 44k
snook_weights = np.array([
    [8, 7, 6, 5, 4],
    [1, 0, 0, 0, 3],
    [2, 0, 0, 0, 2],
    [3, 0, 0, 0, 1],
    [4, 5, 6, 7, 8]
])
snook_weights = 2 ** snook_weights

# 46k (current record)
hsnail_weights = np.array([
    [18, 15, 14, 13, 14],
    [1, 0, 0, 0, 11],
    [2, 0, 0, 0, 10],
    [3, 0, 0, 0, 9],
    [6, 5, 6, 7, 10]
])
hsnail_weights = 2 ** hsnail_weights

# 11k
hsnail2_weights = np.array([
    [17, 15, 14, 13, 13],
    [1, 0, 0, 0, 11],
    [2, 0, 0, 0, 10],
    [3, 0, 0, 0, 9],
    [5, 5, 6, 7, 9]
])
hsnail2_weights = 2 ** hsnail2_weights

# Bad-ish score was forgotten
hsnail3_weights = np.array([
    [20, 18, 17, 16, 15],
    [1, 0, 0, 0, 13],
    [2, 0, 0, 0, 12],
    [3, 0, 0, 0, 11],
    [5, 6, 7, 8, 10]
])
hsnail3_weights = 2 ** hsnail3_weights 

# 15k
diamond_weights = np.array([
    [16, 8, 4, 2, 1],
    [8, 0, 0, 0, 2],
    [4, 0, 0, 0, 4],
    [2, 0, 0, 0, 8],
    [1, 2, 4, 8, 16]
])
diamond_weights = 2 ** diamond_weights

weights = hsnail_weights

a_weights = weights.reshape((25,))
b_weights = np.rot90(weights, 1).reshape((25,))
c_weights = np.rot90(weights, 2).reshape((25,))
d_weights = np.rot90(weights, 3).reshape((25,))
e_weights = np.fliplr(weights).reshape((25,))
f_weights = np.fliplr(np.rot90(weights, 1)).reshape((25,))
g_weights = np.fliplr(np.rot90(weights, 2)).reshape((25,))
h_weights = np.fliplr(np.rot90(weights, 3)).reshape((25,))

def eval_num_moves(game: Gridentify):
    num_ok_moves = 0

    for move in game.valid_moves():
        result = game.board[move[0]] * len(move)
        if result not in good_values:
            continue
        else:
            num_ok_moves += 1

    return num_ok_moves

def board_eval(game: Gridentify):
    board = np.array(game.board)
    # Scrabble eval
    a = np.sum(a_weights * board)
    b = np.sum(b_weights * board)
    c = np.sum(c_weights * board)
    d = np.sum(d_weights * board)
    e = np.sum(e_weights * board)
    f = np.sum(f_weights * board)
    g = np.sum(g_weights * board)
    h = np.sum(h_weights * board)
    scr = max(a, b, c, d, e, f, g, h)

    # Neighbor eval
    nbo = 0
    for list_of_neighbours in game.get_neighbours_of():
        nbo += len(list_of_neighbours)

    return 100 * nbo*np.log10(scr) + scr

def tree_search(game: Gridentify, depth):

    if depth == 0:
        return board_eval(game), None

    else:
        valid_moves = game.valid_moves()

        # return negative infinity if board position has no valid moves.
        if len(valid_moves) == 0:
            return np.NINF, None

        move_evals = np.zeros((len(valid_moves),))

        panic = len(valid_moves) < 5
        # if panic: print('PANIC')

        for i, move in enumerate(valid_moves):
            #print(move)
            # Prune bad moves if not panicing.
            result = game.board[move[0]] * len(move)
            if panic or (len(move) in good_move_lens and result in good_values):
                temp_game = game.copy()
                temp_game.make_move(move)
                move_evals[i], best_move = tree_search(temp_game, depth - 1)
            
            else:
                move_evals[i] = np.NINF

        else:
            move_index = np.argmax(move_evals)
            best_eval = move_evals[move_index]

            return best_eval, valid_moves[move_index]

if __name__ == "__main__":
    # Start a timer.
    start_time = time.time()

    # Make new game.
    test_seed = 20766236554
    # print(f'seed: {test_seed}')
    game = Gridentify(seed=test_seed)
    game.show_board()

    # Initial moves.
    valid_moves = game.valid_moves()

    move_num = 0
    while len(valid_moves) > 0:
        move_num += 1
        print(f'\n--- Move #{move_num} ---')
        print(f'Number of valid moves: {len(valid_moves)}')

        move = []
        while move not in valid_moves:
            # THIS IS WHERE THE MOVE MACHINE GOES.
            num_val_moves = len(valid_moves)
            num_ok_moves = eval_num_moves(game)
            print(f'Number of ok moves: {num_ok_moves}')
            if num_ok_moves > 0:
                a = int(30/num_ok_moves)
                #a = int(num_val_moves/num_ok_moves/4)
                # a = max(0, int(5 - num_ok_moves/5))
                # a = int(100/len(valid_moves))
            else:
                a = 100
            depth = min(a, 4) + 2
            print(f'Depth for next move: {depth}')
            evaluation, move = tree_search(game, depth=depth)
            print(f'Move eval: {evaluation:.2f}')
            #input()

        # Show the game.
        show_move(move)
        game.make_move(move)
        board = np.array(game.board).reshape((5,5))
        print('\n ' + str(board)[1:-1])
        print(f'\nScore: {game.score}')
        # Get new valid moves.
        valid_moves = game.valid_moves()

    print('\nGame Over')

    # End the timer
    end_time = time.time()

    seconds = end_time - start_time
    minutes = seconds // 60
    seconds %= 60
    print(f'Time: {int(minutes)}m {int(seconds)}s')