/**
 * SNAKE GAME ENGINE + AI
 * Browser-based, no dependencies (besides neat.js)
 */

const GRID = 20;         // cells
const CELL = 20;         // px per cell
const CANVAS_SIZE = GRID * CELL;

// Directions
const DIR = { UP: 0, RIGHT: 1, DOWN: 2, LEFT: 3 };
const DX = [0, 1, 0, -1];
const DY = [-1, 0, 1, 0];

// ───────────────────────────────
// SnakeGame — single game instance
// ───────────────────────────────
class SnakeGame {
  constructor() { this.reset(); }

  reset() {
    this.snake = [{ x: 10, y: 10 }];
    this.dir = DIR.RIGHT;
    this.nextDir = DIR.RIGHT;
    this.food = this._placeFood();
    this.score = 0;
    this.steps = 0;
    this.alive = true;
    this.stepsWithoutFood = 0;
  }

  _placeFood() {
    let pos;
    do {
      pos = { x: Math.floor(Math.random() * GRID), y: Math.floor(Math.random() * GRID) };
    } while (this.snake.some(s => s.x === pos.x && s.y === pos.y));
    return pos;
  }

  setDir(d) {
    // Can't reverse directly
    if ((d + 2) % 4 !== this.dir) this.nextDir = d;
  }

  step() {
    if (!this.alive) return;
    this.dir = this.nextDir;
    this.steps++;
    this.stepsWithoutFood++;

    const head = this.snake[0];
    const nx = head.x + DX[this.dir];
    const ny = head.y + DY[this.dir];

    // Wall or self collision
    if (nx < 0 || nx >= GRID || ny < 0 || ny >= GRID ||
        this.snake.some(s => s.x === nx && s.y === ny)) {
      this.alive = false;
      return;
    }

    this.snake.unshift({ x: nx, y: ny });

    if (nx === this.food.x && ny === this.food.y) {
      this.score++;
      this.stepsWithoutFood = 0;
      this.food = this._placeFood();
    } else {
      this.snake.pop();
    }

    // Timeout to avoid infinite loops
    if (this.stepsWithoutFood > GRID * GRID * 2) {
      this.alive = false;
    }
  }

  // ─────────────────────────────────
  // Neural network input features (24 inputs)
  // 8 directions × 3 features (wall dist, body dist, food)
  // ─────────────────────────────────
  getInputs() {
    const head = this.snake[0];
    const inputs = [];

    // 8 directions: N, NE, E, SE, S, SW, W, NW
    const dirs8 = [
      [0, -1], [1, -1], [1, 0], [1, 1],
      [0, 1], [-1, 1], [-1, 0], [-1, -1]
    ];

    for (const [dx, dy] of dirs8) {
      let dist = 0;
      let bodyFound = 0;
      let foodFound = 0;
      let cx = head.x + dx;
      let cy = head.y + dy;

      while (cx >= 0 && cx < GRID && cy >= 0 && cy < GRID) {
        dist++;
        if (!bodyFound && this.snake.slice(1).some(s => s.x === cx && s.y === cy)) {
          bodyFound = 1 / dist;
        }
        if (!foodFound && cx === this.food.x && cy === this.food.y) {
          foodFound = 1;
        }
        cx += dx;
        cy += dy;
      }

      inputs.push(1 / (dist + 1)); // wall proximity
      inputs.push(bodyFound);
      inputs.push(foodFound);
    }

    return inputs; // 24 inputs
  }

  getFitness() {
    // Reward food eaten heavily, small bonus for survival
    return this.score * 100 + Math.min(this.steps, 200);
  }
}

// ───────────────────────────────
// SnakeAI — wraps a game + neural net
// ───────────────────────────────
class SnakeAI {
  constructor(brain) {
    this.brain = brain;
    this.game = new SnakeGame();
    this.fitness = 0;
  }

  think() {
    const inputs = this.game.getInputs();
    const outputs = this.brain.predict(inputs);
    // outputs[0]=UP, [1]=RIGHT, [2]=DOWN, [3]=LEFT
    const maxIdx = outputs.indexOf(Math.max(...outputs));
    this.game.setDir(maxIdx);
    this.game.step();
    this.fitness = this.game.getFitness();
  }

  get alive() { return this.game.alive; }
  get score() { return this.game.score; }
}

// ───────────────────────────────
// SnakeTrainer — manages population training
// ───────────────────────────────
class SnakeTrainer {
  constructor({ populationSize = 100, onGeneration } = {}) {
    this.popSize = populationSize;
    this.onGeneration = onGeneration || (() => {});
    this.ga = new GeneticAlgorithm({
      populationSize,
      eliteCount: 5,
      mutationRate: 0.15,
      mutationStrength: 0.3
    });
    this.ga.init(() => new NeuralNetwork(24, [16, 12], 4));
    this.agents = this._spawnAgents();
    this.running = false;
    this.speed = 1; // steps per frame
    this.generation = 0;
    this.bestScore = 0;
    this.bestEver = 0;
    this.avgScore = 0;
    this.watchIdx = 0; // which agent to render
  }

  _spawnAgents() {
    return this.ga.population.map(brain => new SnakeAI(brain));
  }

  start() { this.running = true; }
  pause() { this.running = false; }
  toggle() { this.running = !this.running; }

  setSpeed(s) { this.speed = Math.max(1, Math.min(200, s)); }

  tick() {
    if (!this.running) return;
    for (let s = 0; s < this.speed; s++) {
      this._stepAll();
    }
  }

  _stepAll() {
    let allDead = true;
    for (let i = 0; i < this.agents.length; i++) {
      if (this.agents[i].alive) {
        this.agents[i].think();
        allDead = false;
      }
    }
    if (allDead) this._nextGen();
  }

  _nextGen() {
    // Report fitnesses
    for (let i = 0; i < this.agents.length; i++) {
      this.ga.setFitness(i, this.agents[i].fitness);
    }

    const scores = this.agents.map(a => a.score);
    this.bestScore = Math.max(...scores);
    this.avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;
    if (this.bestScore > this.bestEver) this.bestEver = this.bestScore;

    this.ga.evolve();
    this.generation = this.ga.generation;
    this.agents = this._spawnAgents();
    this.onGeneration({ generation: this.generation, best: this.bestScore, avg: this.avgScore, bestEver: this.bestEver });
  }

  // Get the agent to visually watch
  getWatchAgent() {
    // Find highest scoring alive agent, or fallback to first
    let best = null;
    for (const a of this.agents) {
      if (a.alive && (!best || a.score > best.score)) best = a;
    }
    return best || this.agents[0];
  }

  getBestBrain() {
    return this.ga.bestEver;
  }
}

// ───────────────────────────────
// SnakeRenderer — draws game to canvas
// ───────────────────────────────
class SnakeRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    canvas.width = CANVAS_SIZE;
    canvas.height = CANVAS_SIZE;
  }

  draw(game, style = 'classic') {
    const ctx = this.ctx;
    ctx.fillStyle = '#080810';
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

    // Grid dots
    ctx.fillStyle = '#1a1a2e';
    for (let x = 0; x < GRID; x++) {
      for (let y = 0; y < GRID; y++) {
        ctx.fillRect(x * CELL + 9, y * CELL + 9, 2, 2);
      }
    }

    // Food
    const fx = game.food.x * CELL;
    const fy = game.food.y * CELL;
    ctx.fillStyle = '#ff6584';
    ctx.beginPath();
    ctx.arc(fx + CELL / 2, fy + CELL / 2, CELL / 2 - 2, 0, Math.PI * 2);
    ctx.fill();

    // Snake
    game.snake.forEach((seg, i) => {
      const t = i / game.snake.length;
      if (i === 0) {
        // Head
        ctx.fillStyle = '#6c63ff';
      } else {
        // Body gradient
        const g = Math.floor(255 * (1 - t * 0.7));
        ctx.fillStyle = `rgb(${Math.floor(80 + t * 20)}, ${Math.floor(g * 0.4)}, ${g})`;
      }
      ctx.fillRect(seg.x * CELL + 1, seg.y * CELL + 1, CELL - 2, CELL - 2);

      // Eyes on head
      if (i === 0) {
        ctx.fillStyle = 'white';
        const eyeOff = { 0: [[-4, -5], [4, -5]], 1: [[5, -4], [5, 4]], 2: [[-4, 5], [4, 5]], 3: [[-5, -4], [-5, 4]] }[game.dir];
        for (const [ex, ey] of eyeOff) {
          ctx.beginPath();
          ctx.arc(seg.x * CELL + CELL / 2 + ex, seg.y * CELL + CELL / 2 + ey, 2.5, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    });

    // Score overlay
    ctx.fillStyle = 'rgba(0,0,0,0.5)';
    ctx.fillRect(4, 4, 70, 22);
    ctx.fillStyle = '#fff';
    ctx.font = '13px monospace';
    ctx.fillText(`Score: ${game.score}`, 8, 20);
  }
}

// ───────────────────────────────
// Mini chart renderer
// ───────────────────────────────
function drawChart(canvas, history) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width = canvas.offsetWidth || 300;
  const H = canvas.height = 120;
  ctx.clearRect(0, 0, W, H);

  if (history.length < 2) return;

  const maxVal = Math.max(...history.map(h => h.best), 1);

  // Grid lines
  ctx.strokeStyle = '#2a2a4a';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = H - (i / 4) * H;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
  }

  const scaleX = W / (history.length - 1);
  const scaleY = H / maxVal;

  // Avg line
  ctx.strokeStyle = '#6c63ff55';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  history.forEach((h, i) => {
    const x = i * scaleX;
    const y = H - h.avg * scaleY;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Best line
  ctx.strokeStyle = '#43e97b';
  ctx.lineWidth = 2;
  ctx.beginPath();
  history.forEach((h, i) => {
    const x = i * scaleX;
    const y = H - h.best * scaleY;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Label
  ctx.fillStyle = '#888899';
  ctx.font = '10px monospace';
  ctx.fillText('best', 4, 12);
  ctx.fillStyle = '#43e97b';
  ctx.fillRect(30, 6, 10, 2);
}

// Export
window.SnakeGame = SnakeGame;
window.SnakeAI = SnakeAI;
window.SnakeTrainer = SnakeTrainer;
window.SnakeRenderer = SnakeRenderer;
window.drawChart = drawChart;
window.CANVAS_SIZE = CANVAS_SIZE;
