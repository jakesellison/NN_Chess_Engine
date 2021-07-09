from data_utils import get_board_state
import chess
import chess.pgn
import keras
import numpy as np
import matplotlib.pyplot as plt
from data_utils import get_board_state
from data_utils import get_legal_moves
from data_utils import gen_mappings

SaP_white_early_v1 = keras.models.load_model('keras_models/SaP_white_early')
SaP_white_mid_v1 = keras.models.load_model('keras_models/SaP_white_mid')
SaP_white_late_v1 = keras.models.load_model('keras_models/SaP_white_late')
MaP_white_early_v1 = keras.models.load_model('keras_models/MaP_white_early')
MaP_white_mid_v1 = keras.models.load_model('keras_models/MaP_white_mid')
MaP_white_late_v1 = keras.models.load_model('keras_models/MaP_white_late')

SaP_white_early_v2 = keras.models.load_model('keras_models/v2/SaP_white_early')
SaP_white_mid_v2 = keras.models.load_model('keras_models/v2/SaP_white_mid')
SaP_white_late_v2 = keras.models.load_model('keras_models/v2/SaP_white_late')
MaP_white_pawn_v2 = keras.models.load_model('keras_models/v2/MaP_white_pawn')
MaP_white_knight_v2 = keras.models.load_model('keras_models/v2/MaP_white_knight')
MaP_white_bishop_v2 = keras.models.load_model('keras_models/v2/MaP_white_bishop')
MaP_white_rook_v2 = keras.models.load_model('keras_models/v2/MaP_white_rook')
MaP_white_queen_v2 = keras.models.load_model('keras_models/v2/MaP_white_queen')
MaP_white_king_v2 = keras.models.load_model('keras_models/v2/MaP_white_king')

def model_get_move_v1(board, heatmap = False):
    
    X_SaP = get_board_state(board) + [board.ply()] + [1]
    SaP_white_early_v1.predict([X_SaP])

    # Get the initial prediction from the SaP model
    if board.ply() <= 14:
        from_square_grid = SaP_white_early_v1.predict([X_SaP]).flatten()
    elif 15 <= board.ply() <= 40:
        from_square_grid = SaP_white_mid_v1.predict([X_SaP]).flatten()
    else: 
        from_square_grid = SaP_white_late_v1.predict([X_SaP]).flatten()
  
    # Filter that prediction to find the top valid selection
    temp_from_square_grid = from_square_grid
    for i in range (64): # go through each of the probabilities found by the NN
        valid = False
        from_square = np.argmax(temp_from_square_grid.flatten())
        for j in range(64): # using the found from-square, see if there is a valid move originating from that square
            try:
                board.find_move(from_square,j)
                valid = True
                break
            except:
                continue
        if valid == True:
            break
        else:
            temp_from_square_grid[from_square] = 0
    
    if heatmap == True:
        from_square_grid = np.flipud(from_square_grid.reshape(8, 8))
        plt.imshow(from_square_grid, cmap='hot', interpolation='nearest')
        print('Origin Square:')
        plt.show()
    
    X_MaP = get_board_state(board) + [board.ply()] + [1] + [from_square]
    
    if board.ply() <= 14:
        to_square_grid = MaP_white_early_v1.predict([X_MaP]).flatten()
    elif 15 <= board.ply() <= 40:
        to_square_grid = MaP_white_mid_v1.predict([X_MaP]).flatten()
    else: 
        to_square_grid = MaP_white_late_v1.predict([X_MaP]).flatten()

    # Filter that prediction to find the top valid selection
    for i in range(64): # go through each of the probabilities found by the NN
        to_square = np.argmax(to_square_grid)
        try:
            board.find_move(from_square,to_square)
            break
        except:
            to_square_grid[to_square] = 0
            
    if heatmap == True:
        to_square_grid = np.flipud(to_square_grid.reshape(8, 8))
        plt.imshow(to_square_grid, cmap='hot', interpolation='nearest')
        print('Destination Square:')
        plt.show()
    
    return(board.find_move(from_square, to_square))
        
def model_get_move_v2(board, heatmap = False):
    
    white_check_map_tmp, black_check_map_tmp, white_cap_map_tmp, black_cap_map_tmp = gen_mappings(board)
    X_SaP = get_board_state(board) + white_check_map_tmp + black_check_map_tmp + white_cap_map_tmp + black_cap_map_tmp

    # Get the initial prediction from the SaP model
    if board.ply() <= 14:
        from_square_grid = SaP_white_early_v2.predict([X_SaP]).flatten()
    elif 15 <= board.ply() <= 40:
        from_square_grid = SaP_white_mid_v2.predict([X_SaP]).flatten()
    else: 
        from_square_grid = SaP_white_late_v2.predict([X_SaP]).flatten()
  
    # Filter that prediction to find the top valid selection
    temp_from_square_grid = from_square_grid
    for i in range (64): # go through each of the probabilities found by the NN
        valid = False
        from_square = np.argmax(temp_from_square_grid.flatten())
        for j in range(64): # using the found from-square, see if there is a valid move originating from that square
            try:
                board.find_move(from_square,j)
                valid = True
                break
            except:
                continue
        if valid == True:
            break
        else:
            temp_from_square_grid[from_square] = 0
    
    if heatmap == True:
        from_square_grid = np.flipud(from_square_grid.reshape(8, 8))
        plt.imshow(from_square_grid, cmap='hot', interpolation='nearest')
        print('Origin Square:')
        plt.show()
    
    X_MaP = get_board_state(board) + white_check_map_tmp + black_check_map_tmp + white_cap_map_tmp + black_cap_map_tmp + get_legal_moves(board, from_square)
    
    # Get the initial prediction from the MaP model
    if board.piece_at(from_square) == 'P':
        to_square_grid = MaP_white_pawn_v2.predict([X_MaP]).flatten()
    elif board.piece_at(from_square) == 'N':
        to_square_grid = MaP_white_knight_v2.predict([X_MaP]).flatten()
    elif board.piece_at(from_square) == 'B':
        to_square_grid = MaP_white_bishop_v2.predict([X_MaP]).flatten()
    elif board.piece_at(from_square) == 'R':
        to_square_grid = MaP_white_rook_v2.predict([X_MaP]).flatten()
    elif board.piece_at(from_square) == 'Q':
        to_square_grid = MaP_white_queen_v2.predict([X_MaP]).flatten()
    else:
        to_square_grid = MaP_white_king_v2.predict([X_MaP]).flatten()

    # Filter that prediction to find the top valid selection
    for i in range(64): # go through each of the probabilities found by the NN
        to_square = np.argmax(to_square_grid)
        try:
            board.find_move(from_square,to_square)
            break
        except:
            to_square_grid[to_square] = 0
            
    if heatmap == True:
        to_square_grid = np.flipud(to_square_grid.reshape(8, 8))
        plt.imshow(to_square_grid, cmap='hot', interpolation='nearest')
        print('Destination Square:')
        plt.show()
    
    return(board.find_move(from_square, to_square))

def rand_get_move(board):
    i = 0
    for move in board.legal_moves:
        i += 1
    rand_int = np.random.randint(low=0, high=i, size=1)[0]
    i = 0
    for move in board.legal_moves:
        if i == rand_int:
            return(move)
        i += 1