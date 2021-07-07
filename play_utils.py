from data_utils import get_board_state
import chess
import chess.pgn
import keras
import numpy as np
import matplotlib.pyplot as plt

SaP_white_early = keras.models.load_model('keras_models/SaP_white_early')
SaP_white_mid = keras.models.load_model('keras_models/SaP_white_mid')
SaP_white_late = keras.models.load_model('keras_models/SaP_white_late')
MaP_white_early = keras.models.load_model('keras_models/MaP_white_early')
MaP_white_mid = keras.models.load_model('keras_models/MaP_white_mid')
MaP_white_late = keras.models.load_model('keras_models/MaP_white_late')

def model_get_move(board, winner, heatmap = False):
    
    input_data = get_board_state(board) + winner
    
    # Get the initial prediction from the SaP model
    if board.ply() <= 14:
        from_square_grid = SaP_white_early.predict([input_data]).flatten()
    elif 15 <= board.ply() <= 40:
        from_square_grid = SaP_white_mid.predict([input_data]).flatten()
    else: 
        from_square_grid = SaP_white_late.predict([input_data]).flatten()
  
    # Filter that prediction to find the top valid selection
    temp_from_square_grid = from_square_grid
    for i in range (63): # go through each of the probabilities found by the NN
        valid = False
        from_square = np.argmax(temp_from_square_grid.flatten())
        for j in range(63): # using the found from-square, see if there is a valid move originating from that square
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
        plt.show()
    
    # Get the initial prediction from the MaP model
    if board.ply() <= 14:
        to_square_grid = MaP_white_early.predict([input_data + [from_square]]).flatten()
    elif 15 <= board.ply() <= 40:
        to_square_grid = MaP_white_mid.predict([input_data + [from_square]]).flatten()
    else: 
        to_square_grid = MaP_white_late.predict([input_data + [from_square]]).flatten()

    # Filter that prediction to find the top valid selection
    for i in range (63): # go through each of the probabilities found by the NN
        to_square = np.argmax(to_square_grid)
        try:
            board.find_move(from_square,to_square)
            break
        except:
            to_square_grid[to_square] = 0
    
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