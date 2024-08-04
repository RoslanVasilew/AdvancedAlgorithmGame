
# Advanced Algorithm Game

Welcome to the Advanced Algorithm Game! This is a strategic evasion game where you must navigate your player piece across a chessboard while avoiding enemy pieces controlled by advanced algorithms. This project utilizes the alpha-beta pruning algorithm for AI decision-making.

## Features

- **Dynamic Difficulty Levels**: Easy, Medium, and Hard modes.
- **Special Abilities**: Use abilities like Freeze, Push, Double Move, and Obstacle Shift to outsmart your enemies.
- **AI-Controlled Enemies**: Chess pieces with directional and diagonal movement patterns.
- **Moving Train**: A train obstacle that adds an extra layer of challenge.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/RoslanVasilew/AdvancedAlgorithmGame.git
   cd AdvancedAlgorithmGame
   ```

2. **Install Pygame:**
   ```bash
   pip install pygame
   ```

3. **Download Assets:**
   Ensure you have the following assets in the `Assets` directory:
   - `mighty-force.png`
   - `ice-cube.png`
   - `earth-spit.png`
   - `kangaroo.png`
   - `ram.svg`
   - `warlord-helmet.svg`
   - `crown.png`
   - `steam-locomotive.png`

4. **Run the game:**
   ```bash
   python chess_evasion_game.py
   ```

## Gameplay

### Objective

Navigate your player piece from the top row to the bottom row of the chessboard while avoiding enemy pieces and obstacles.

### Controls

- **Movement**: Use `W`, `A`, `S`, `D` keys to move Up, Left, Down, and Right respectively. Diagonal movements can be made with `Q`, `E`, `Z`, `C` keys.
- **Abilities**: Abilities are triggered using their respective keys (`R`, `T`, `Y`, `U`). You can also click on the ability buttons at the bottom of the screen.

### Abilities

- **Freeze (R)**: Freezes enemy pieces within a 2-block radius for 2 turns.
- **Push (T)**: Pushes enemy pieces away from the player.
- **Double Move (Y)**: Allows the player to move twice in one turn.
- **Obstacle Shift (U)**: Respawns the obstacles on the board.

### Difficulty Levels

- **Easy**: Fewer enemies and no train.
- **Medium**: More enemies and the train obstacle.
- **Hard**: Maximum enemies and the train obstacle, with added dynamic obstacles.

## Code Structure

- **`chess_evasion_game.py`**: Main game file containing all classes and game logic.
- **`Assets/`**: Directory containing all game assets (images and icons).

### Classes

- **Player**: Handles player movement and abilities.
- **ChessPiece**: Represents enemy chess pieces with movement logic.
- **Obstacle**: Represents static obstacles on the board.
- **Train**: Represents a moving train obstacle on the board.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes. Ensure your code adheres to the project's coding standards.

## Acknowledgements

- Pygame for the game development library.
- Contributors for their help in development.

---

Enjoy the game!
