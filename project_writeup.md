# Neural Network Chess Engine in Python
Jacob Ellison

## Abstract
My goal for this project was to build a chess engine on a sequential neural network. I knew that the engine would play poorly due to the time constraints of the project and the complex nature of predicting a chess move, so my goal was to make an engine that would could generate and play legal moves, and any real performance would be a bonus over that baseline. I delegated the decision-making for this engine to a series of neural nets devoted to specific aspects of choosing a chess move (e.g., which piece to move, where to move it to, which color is playing, etc). I used data from the [FICS chess games database](https://www.ficsgames.org/download.html), which contains millions of games played by 2000+ ELO rated players, and I processed this data using the [Python Chess](https://python-chess.readthedocs.io/en/latest/) library.

## Design
At a macro level, I wanted to create a network that would take a board state as input and use the move that the player selected as the target. However, this selection becomes complicated quickly: The neural network has no innate conception of which pieces belong to it, or what constitutes a legal move. For this reason, for any given board state on the 8x8 chess board, the number of possible moves must be represented by the total number of squares that could contain a piece (64) * the total number of squares that a piece could move to (64), yielding a total decision space of 4,096 moves.

I decided to simplify and optimize this decision-making process by delegating selecting a piece (SaP) and moving a piece (MaP) to separate network. Done in this way, the input to each network is the numeric representation of the board state and the target is a 64-length array where the chosen square (either the origin square for SaP or the destination square for MaP) is given by a 1 and the other squares are marked with 0s. 

I elected to further narrow the scope of the decisions in this chess engine by breaking out the SaP and MaP models into sub-neural networks that operate on specific board-states. I created separate networks for the white and black pieces (because I found that the models would often try to play with the opponent's pieces). I then split these networks again based on turn count: The early game in chess is somewhat formulaic with known moves and countermoves, and I wanted the network to memorize these patterns. In the late game, however, the board becomes much more positional and the squares commonly used in the early game are often unoccupied. For these reasons, I divided the training data up into early, mid, and late-stage boards and fit models to play on each.

## Data
The FICS database contains chess games represented in PGN form. PGN is a standard chess notation where each move is represented by an alphanumeric combination. The initial letter represents the piece being moved (r = rook, b = bishop, etc.), and the following letter-number combination shows the destination for that piece. Chess rows are called 'ranks' and are numbered, and chess columns are called 'files' and are lettered. The move ra8 is read as the rook moving to the square A8. This notation uses special characters to show certain situations like checks and castling. Below is an example of a PGN string:

`<1. e4 e6 2. d4 d5 3. Nc3 Nf6 4. Bg5 dxe4 5. Nxe4 Nbd7 6. Bd3 h6 7. Bh4 g5 8. Bg3 Nxe4 9. Bxe4 f5 10. Qh5+ Ke7 11. h4 {White resigns} 0-1>`

I converted the PGN data into a series of board states. The eleven-move game in the PGN above would be converted to 21 board states, and the targets for each board are the succeeding move's origin and destination squares. I saved these boards using a modified [FEN notation](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation). FEN strings still require special characters and letters, so converted these to numbers and concatenated on additional metadata about the board like whose turn it is and castling rights.

## Process

*Models*
  
I used 0.1 dropout on each of my dense layers to prevent overfitting, especially because many of my datasets became relatively small after I had filtered my training data down to the correct scope for each model (for instance, a SaP model for the white pieces in the early game). I used ReLU as my activation for the hidden layers and then a softmax activation in my output layer to show predicted probabilities within the 64-square decision space. 

I also began creating a new series of MaP models that operate on specific pieces: For instance, if the SaP model chooses a pawn, then then MaP decision is passed to a model trained specifically on pawn moves. I found that the chance of overfitting rose dramatically with this approach. I also increased the size of the input considerably, which caused the data-processing and model-training times to spike, and as a result, this stage is still a work in progress.

## Results

Below is the evaluation for each NN:

| Model    | Game Stage  | Loss   | Accuracy |
| -------- | ----------- | ------ | -------- |
| SaP      | Early       | 1.15   |   0.61   |
| SaP      | Mid         | 1.98   |   0.34   |
| SaP      | Late        | 1.61   |   0.40   |
| MaP      | Early       | 0.32   |   0.89   |
| MaP      | Mid         | 1.74   |   0.47   |
| MaP      | Late        | 2.90   |   0.19   |

## Tools
- Python chess for move validation, data processing, and board visualizations
- Numpyfor data manipulation
- Keras for modeling
- Matplotlib for plotting move heatmaps

**Next Steps**
1. I built this model using only dense layers, however I'd like to try adding convolutional layers to see if they will help the engine see useful move patterns anywhere on the board (like finding an easy queen capture anywhere on the board).
2. Currently I move from early -> mid -> late game by tracking turn count. Instead, I'd like to try advancing the game stage based on pieces remaining on the board. 
3. I would like to try splitting out the MaP neural networks into separate models for each piece. The SaP model might choose a pawn move, for instance, and I'll then pass the MaP decision to a NN built with only pawn-move training data. I think that this change might reduce the amount of piece-shuffling in the late game.
4. Lastly, and a larger lift than the preceding points is redesigning the chess engine to use Q-learning to build a new model that would better understand piece evaluations and have a sense for how a move will impact future outcomes, which would perform much better than my current near-sighted implementation.