## Project Proposal

### Purpose:
I will create a chess engine using a neural network. I will provide a board state (a snapshot of a position within a chess game) as input and get as output the move predicted by the neural network. I will assess this model by comparing the performance of my engine against moves that are generated randomly.

### Data Description:
To create the neural network, I will need to convert the board state into numeric inputs. The chess database I'm using provides data in PGN form. PGN is a standard chess notation where each move is represented by an alphanumeric combination. The initial letter represents the piece being moved (r = rook, b = bishop, etc.), and the following letter-number combination shows the destination for that piece. Chess rows are called 'ranks' and are numbered, and chess columns are called 'files' and are lettered. The move ra8 is read as the rook moving to the square A8. This notation uses special characters to show certain situations like checks and castling. Below is an example of a PGN string:

`<1. e4 e6 2. d4 d5 3. Nc3 Nf6 4. Bg5 dxe4 5. Nxe4 Nbd7 6. Bd3 h6 7. Bh4 g5 8. Bg3 Nxe4 9. Bxe4 f5 10. Qh5+ Ke7 11. h4 {White resigns} 0-1>`

FICS Games Database has millions of games in PGN format https://www.ficsgames.org/

I will convert the PGN data into a series of board states. The eleven-move game in the PGN above would be converted to eleven board states where each board state shows the result of the preceding moves. I will save these board states using the [FEN string notation](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation). FEN strings still require special characters and letters, so I will need to convert these to numbers. I will experiment with a few different methods for rendering and processing these boards:
* Store the FEN string as a numeric vector. Allow for 64 inputs to NN (1 for each piece on the board), plus some additional inputs for board metadata (e.g., whether the king has castled)
* Store the board state as a matrix and use a convolutional neural network instead of a sequential network. Treat the board more like image data to form a prediction.

### Tools & Process Outline
In addition to the database mentioned above, I will be using the python-chess library, which provides convenience functions to convert PGN to FEN, keep track of board states, and test for valid moves.

Part of the challenge of this project will be to create a NN that outputs valid moves. There are plenty of illegal moves in chess--for instance, moving a piece through another piece or not protecting the king when it is in check. To avoid situations where my model suggests only illegal moves, I will filter the decision making process through two neural nets.

* The first NN takes the board state as input and outputs the piece to move: one output neuron for each piece and a softmax activation function in the final layer to assign probabilities to each piece. Outside of the NN, I will choose the piece with the highest probability. If moving that piece is impossible (for instance, if it has been captured and is no longer in the game), I will eliminate that choice and look at the next highest probability and repeat until I find a valid piece.
* The second NN will take as input the board state as well as an encoding that represents the position of the selected piece (from the previous NN) and will output a number corresponding to the square to move the piece to, perhaps as a int 1-64 that describes where on the 8x8 board to move the piece. Again, I will use softmax on the output and then go through the top moves until I find a valid move.
* I'm sure the implementation will change significantly as I get into testing, but this should serve as a roadmap.
