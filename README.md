# 🤖 AI Game World

> Watch neural networks evolve and learn to play games — in real time, right in your browser.

**Live demo:** (deploy via GitHub Pages)

## What is this?

AI Game World is a browser-based platform where AI agents train themselves to play games using neural networks and genetic algorithms. No server required — everything runs in pure JavaScript.

Every week, a new game is added. Each game comes with:
- A fully playable browser game (you can play too!)
- A **Master AI** that trains continuously and improves over time
- The ability to **spawn new AI agents** and watch them learn from scratch
- Real-time training visualizations (population health, fitness charts)

## 🎮 Games

| Game | Status | AI Method |
|------|--------|-----------|
| 🐍 Snake | ✅ Live | Genetic Algorithm + Neural Net (24 inputs, 8-direction vision) |
| 🔴 Connect 4 | ✅ Live | GA + Neural Net vs Minimax opponents |
| 🏓 Pong | 🔜 Soon | Two AIs vs each other |
| 🚗 Car Racing | 🔜 Soon | NEAT on procedural track |
| 🧩 Tetris | 🔜 Soon | GA fitness on line clears |

## 🧠 How it works

### Neural Network
Each agent has a small feedforward neural network:
- **Snake:** 24 inputs (8 directions × 3 features: wall, body, food) → 16→12 hidden → 4 outputs (direction)
- **Connect 4:** 43 inputs (board state) → 32→16 hidden → 7 outputs (column choice)

### Genetic Algorithm
1. Start with a random population (50–100 agents)
2. Each agent plays the game until it dies or wins
3. Fitness score assigned based on performance
4. Top agents survive (elitism), rest are bred via crossover + mutation
5. Repeat from step 2

### Training Modes
- **Master AI**: always-on, continuously improving across sessions (saved to localStorage)
- **Custom agents**: spawn your own with configurable population size, starting from scratch or seeded from the Master AI

## 📁 Project Structure

```
ai-game-world/
├── index.html              # Main homepage
├── css/
│   └── style.css           # Global dark theme styles
├── shared/
│   └── neat.js             # Neural network + genetic algorithm engine
└── games/
    ├── snake/
    │   ├── index.html      # Snake game page
    │   └── snake.js        # Snake engine + AI trainer
    └── connect4/
        ├── index.html      # Connect 4 game page
        └── connect4.js     # Connect 4 engine + AI trainer
```

## 🚀 Getting Started

### Play locally
Just open `index.html` in your browser. No build step, no dependencies.

### Deploy to GitHub Pages
1. Fork this repo
2. Go to Settings → Pages
3. Set source to `main` branch, root folder
4. Done — your site is live at `https://yourusername.github.io/AI-game-world`

## 🤝 Contributing

Want to add a game? Here's the template:

1. Create `games/yourgame/` folder
2. Add `index.html` and `yourgame.js`
3. Implement the game engine with a `getInputs()` method
4. Use the shared `NeuralNetwork` + `GeneticAlgorithm` from `shared/neat.js`
5. Add your game card to `index.html`

See `games/snake/snake.js` for a full example.

## 📜 License

MIT — do whatever you want with it.
