/**
 * SNAKE — Game Engine + AI
 * Neural net trained via Genetic Algorithm
 */

const GRID = 20;
const CELL = 20;
const CANVAS_SIZE = GRID * CELL;

const DIR = { UP: 0, RIGHT: 1, DOWN: 2, LEFT: 3 };
const DX  = [0, 1, 0, -1];
const DY  = [-1, 0, 1, 0];

// ─────────────────────────────────────
// SnakeGame
// ─────────────────────────────────────
class SnakeGame {
  constructor() { this.reset(); }

  reset() {
    this.snake = [{ x: 10, y: 10 }];
    this.dir   = DIR.RIGHT;
    this.nextDir = DIR.RIGHT;
    this.food  = this._placeFood();
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
    if ((d + 2) % 4 !== this.dir) this.nextDir = d;
  }

  step() {
    if (!this.alive) return;
    this.dir = this.nextDir;
    this.steps++;
    this.stepsWithoutFood++;

    const head = this.snake[0];
    const nx   = head.x + DX[this.dir];
    const ny   = head.y + DY[this.dir];

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

    if (this.stepsWithoutFood > GRID * GRID * 2) this.alive = false;
  }

  // 24 inputs: 8 directions x (wall proximity, body proximity, food)
  getInputs() {
    const head = this.snake[0];
    const inputs = [];
    const dirs8 = [[0,-1],[1,-1],[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1]];
    for (const [dx, dy] of dirs8) {
      let dist = 0, bodyFound = 0, foodFound = 0;
      let cx = head.x + dx, cy = head.y + dy;
      while (cx >= 0 && cx < GRID && cy >= 0 && cy < GRID) {
        dist++;
        if (!bodyFound && this.snake.slice(1).some(s => s.x === cx && s.y === cy)) bodyFound = 1 / dist;
        if (!foodFound && cx === this.food.x && cy === this.food.y) foodFound = 1;
        cx += dx; cy += dy;
      }
      inputs.push(1 / (dist + 1));
      inputs.push(bodyFound);
      inputs.push(foodFound);
    }
    return inputs;
  }

  getFitness() {
    return this.score * 100 + Math.min(this.steps, 200);
  }
}

// ─────────────────────────────────────
// SnakeAIAgent — one agent with a brain
// ─────────────────────────────────────
class SnakeAIAgent {
  constructor(brain) {
    this.brain   = brain;
    this.game    = new SnakeGame();
    this.fitness = 0;
  }

  think() {
    const inputs  = this.game.getInputs();
    const outputs = this.brain.predict(inputs);
    this.game.setDir(outputs.indexOf(Math.max(...outputs)));
    this.game.step();
    this.fitness = this.game.getFitness();
  }

  get alive() { return this.game.alive; }
  get score() { return this.game.score; }
}

// ─────────────────────────────────────
// SnakeAIInstance — full trainable AI
// One per "agent slot" (master or custom)
// ─────────────────────────────────────
class SnakeAIInstance {
  constructor({ name = 'AI', populationSize = 100, isMaster = false, onGeneration } = {}) {
    this.name        = name;
    this.isMaster    = isMaster;
    this.popSize     = populationSize;
    this.onGeneration = onGeneration || (() => {});
    this.generation  = 0;
    this.bestEver    = 0;
    this.lastBest    = 0;
    this.lastAvg     = 0;
    this.running     = false;
    this._rafId      = null;

    this.ga = new GeneticAlgorithm({
      populationSize,
      eliteCount: 5,
      mutationRate: 0.15,
      mutationStrength: 0.3,
    });
    this.ga.init(() => new NeuralNetwork(24, [16, 12], 4));

    this._agents = this._spawnAgents();
    this._speed  = 10; // steps per animation frame for the watch agent
    this._allDead = false;

    // Watch agent — the best alive one, rendered to canvas
    this._watchAgentIdx = 0;
  }

  _spawnAgents() {
    return this.ga.population.map(brain => new SnakeAIAgent(brain));
  }

  // Called every animation frame by the renderer
  tick() {
    if (!this.running) return;

    // Run _speed steps for all agents
    for (let s = 0; s < this._speed; s++) {
      let anyAlive = false;
      for (const agent of this._agents) {
        if (agent.alive) { agent.think(); anyAlive = true; }
      }
      if (!anyAlive) { this._nextGen(); break; }
    }
  }

  _nextGen() {
    const fitnesses = this._agents.map(a => a.fitness);
    fitnesses.forEach((f, i) => this.ga.setFitness(i, f));

    const scores = this._agents.map(a => a.score);
    this.lastBest = Math.max(...scores);
    this.lastAvg  = scores.reduce((a, b) => a + b, 0) / scores.length;
    if (this.lastBest > this.bestEver) this.bestEver = this.lastBest;

    this.ga.evolve();
    this.generation = this.ga.generation;
    this._agents = this._spawnAgents();

    this.onGeneration({
      generation: this.generation,
      best: this.lastBest,
      avg:  this.lastAvg,
      bestEver: this.bestEver,
    });
  }

  // Best alive agent for rendering
  getWatchAgent() {
    let best = null;
    for (const a of this._agents) {
      if (a.alive && (!best || a.score > best.score)) best = a;
    }
    return best || this._agents[0];
  }

  getAliveCount() {
    return this._agents.filter(a => a.alive).length;
  }

  start()      { this.running = true; }
  pause()      { this.running = false; }
  toggle()     { this.running = !this.running; }
  setSpeed(s)  { this._speed = Math.max(1, Math.min(300, s)); }
  getBestBrain() { return this.ga.bestEver; }
}

// ─────────────────────────────────────
// SnakeRenderer — draws to canvas
// ─────────────────────────────────────
class SnakeRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx    = canvas.getContext('2d');
    canvas.width  = CANVAS_SIZE;
    canvas.height = CANVAS_SIZE;
  }

  draw(game) {
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
    ctx.fillStyle = '#ff6584';
    ctx.beginPath();
    ctx.arc(game.food.x * CELL + CELL / 2, game.food.y * CELL + CELL / 2, CELL / 2 - 2, 0, Math.PI * 2);
    ctx.fill();

    // Snake
    game.snake.forEach((seg, i) => {
      const t = i / game.snake.length;
      ctx.fillStyle = i === 0
        ? '#6c63ff'
        : `rgb(${Math.floor(80 + t * 20)}, ${Math.floor(255 * (1 - t * 0.7) * 0.4)}, ${Math.floor(255 * (1 - t * 0.7))})`;
      ctx.fillRect(seg.x * CELL + 1, seg.y * CELL + 1, CELL - 2, CELL - 2);

      if (i === 0) {
        ctx.fillStyle = 'white';
        const eyeOff = { 0: [[-4,-5],[4,-5]], 1: [[5,-4],[5,4]], 2: [[-4,5],[4,5]], 3: [[-5,-4],[-5,4]] }[game.dir];
        for (const [ex, ey] of eyeOff) {
          ctx.beginPath();
          ctx.arc(seg.x * CELL + CELL / 2 + ex, seg.y * CELL + CELL / 2 + ey, 2.5, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    });

    // Score overlay
    ctx.fillStyle = 'rgba(0,0,0,0.5)';
    ctx.fillRect(4, 4, 72, 22);
    ctx.fillStyle = '#fff';
    ctx.font = '13px monospace';
    ctx.fillText('Score: ' + game.score, 8, 20);
  }
}

// Population dot grid helper
function buildPopGrid(containerId, size) {
  const el = document.getElementById(containerId);
  if (!el) return;
  el.innerHTML = '';
  for (let i = 0; i < size; i++) {
    const d = document.createElement('div');
    d.className = 'pop-dot';
    d.id = containerId + '-dot-' + i;
    el.appendChild(d);
  }
}

function updatePopGrid(containerId, agents) {
  const best = agents.reduce((b, a) => (!b || (a.alive && a.score > b.score)) ? a : b, null);
  agents.forEach((a, i) => {
    const dot = document.getElementById(containerId + '-dot-' + i);
    if (!dot) return;
    dot.className = 'pop-dot' + (a === best && a.alive ? ' best' : a.alive ? ' alive' : '');
  });
}

// Chart
function drawChart(canvas, history) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width = canvas.offsetWidth || 300;
  const H = canvas.height = 120;
  ctx.clearRect(0, 0, W, H);
  if (history.length < 2) return;

  const maxVal = Math.max(...history.map(h => h.best), 1);
  ctx.strokeStyle = '#2a2a4a'; ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = H - (i / 4) * H;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
  }

  const draw = (data, color, lineWidth) => {
    ctx.strokeStyle = color; ctx.lineWidth = lineWidth;
    ctx.beginPath();
    data.forEach((h, i) => {
      const x = i * W / (data.length - 1);
      const y = H - h.best * (H / maxVal);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
  };

  draw(history, '#6c63ff55', 1.5);
  draw(history.map(h => ({ best: h.avg })), '#6c63ff', 1.5);
  draw(history, '#43e97b', 2);
}

// Exports
window.SnakeGame        = SnakeGame;
window.SnakeAIAgent     = SnakeAIAgent;
window.SnakeAIInstance  = SnakeAIInstance;
window.SnakeRenderer    = SnakeRenderer;
window.buildPopGrid     = buildPopGrid;
window.updatePopGrid    = updatePopGrid;
window.drawChart        = drawChart;
window.CANVAS_SIZE      = CANVAS_SIZE;
window.GRID             = GRID;
