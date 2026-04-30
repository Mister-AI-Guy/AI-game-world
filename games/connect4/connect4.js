/**
 * CONNECT 4 — Game Engine + AI
 * Neural net trained via Genetic Algorithm against Minimax opponents
 */

const C4_COLS = 7;
const C4_ROWS = 6;
const C4_CELL = 64;

// ─────────────────────────────────────
// Connect4Game
// ─────────────────────────────────────
class Connect4Game {
  constructor() { this.reset(); }

  reset() {
    this.board = Array.from({ length: C4_ROWS }, () => new Array(C4_COLS).fill(0));
    this.current = 1;
    this.winner = 0;
    this.over = false;
    this.lastMove = null;
    this.winLine = null;
    this.moveCount = 0;
  }

  getValidCols() {
    return Array.from({ length: C4_COLS }, (_, i) => i).filter(c => this.board[0][c] === 0);
  }

  drop(col) {
    if (this.over || col < 0 || col >= C4_COLS || this.board[0][col] !== 0) return false;
    for (let row = C4_ROWS - 1; row >= 0; row--) {
      if (this.board[row][col] === 0) {
        this.board[row][col] = this.current;
        this.lastMove = { row, col };
        this.moveCount++;
        const win = this.checkWin(row, col, this.current);
        if (win) {
          this.winner = this.current;
          this.winLine = win;
          this.over = true;
        } else if (this.moveCount === C4_ROWS * C4_COLS) {
          this.over = true;
        } else {
          this.current = this.current === 1 ? 2 : 1;
        }
        return true;
      }
    }
    return false;
  }

  clone() {
    const g = new Connect4Game();
    g.board = this.board.map(r => [...r]);
    g.current = this.current;
    g.winner = this.winner;
    g.over = this.over;
    g.lastMove = this.lastMove ? { ...this.lastMove } : null;
    g.moveCount = this.moveCount;
    g.winLine = this.winLine ? this.winLine.map(x => [...x]) : null;
    return g;
  }

  checkWin(row, col, player) {
    const dirs = [[0, 1], [1, 0], [1, 1], [1, -1]];
    for (const [dr, dc] of dirs) {
      const line = [[row, col]];
      for (let d = 1; d < 4; d++) {
        const r = row + dr * d, c = col + dc * d;
        if (r >= 0 && r < C4_ROWS && c >= 0 && c < C4_COLS && this.board[r][c] === player) line.push([r, c]);
        else break;
      }
      for (let d = 1; d < 4; d++) {
        const r = row - dr * d, c = col - dc * d;
        if (r >= 0 && r < C4_ROWS && c >= 0 && c < C4_COLS && this.board[r][c] === player) line.push([r, c]);
        else break;
      }
      if (line.length >= 4) return line.slice(0, 4);
    }
    return null;
  }

  // 43 inputs: board from AI perspective + current turn flag
  getInputs(forPlayer) {
    const inputs = [];
    for (let r = 0; r < C4_ROWS; r++) {
      for (let c = 0; c < C4_COLS; c++) {
        const cell = this.board[r][c];
        inputs.push(cell === forPlayer ? 1 : cell === 0 ? 0 : -1);
      }
    }
    inputs.push(this.current === forPlayer ? 1 : -1);
    return inputs;
  }

  scorePosition(player) {
    let score = 0;
    const opp = player === 1 ? 2 : 1;
    const center = this.board.map(r => r[3]);
    score += center.filter(c => c === player).length * 3;
    for (let r = 0; r < C4_ROWS; r++) {
      for (let c = 0; c <= C4_COLS - 4; c++) {
        const w = [this.board[r][c], this.board[r][c+1], this.board[r][c+2], this.board[r][c+3]];
        score += this._scoreWindow(w, player, opp);
      }
    }
    for (let c = 0; c < C4_COLS; c++) {
      for (let r = 0; r <= C4_ROWS - 4; r++) {
        const w = [this.board[r][c], this.board[r+1][c], this.board[r+2][c], this.board[r+3][c]];
        score += this._scoreWindow(w, player, opp);
      }
    }
    for (let r = 0; r <= C4_ROWS - 4; r++) {
      for (let c = 0; c <= C4_COLS - 4; c++) {
        const w1 = [this.board[r][c], this.board[r+1][c+1], this.board[r+2][c+2], this.board[r+3][c+3]];
        const w2 = [this.board[r+3][c], this.board[r+2][c+1], this.board[r+1][c+2], this.board[r][c+3]];
        score += this._scoreWindow(w1, player, opp);
        score += this._scoreWindow(w2, player, opp);
      }
    }
    return score;
  }

  _scoreWindow(w, p, o) {
    const pc = w.filter(x => x === p).length;
    const oc = w.filter(x => x === o).length;
    const ec = w.filter(x => x === 0).length;
    if (pc === 4) return 100;
    if (pc === 3 && ec === 1) return 5;
    if (pc === 2 && ec === 2) return 2;
    if (oc === 3 && ec === 1) return -4;
    return 0;
  }
}

// ─────────────────────────────────────
// Minimax opponent
// ─────────────────────────────────────
class Connect4Minimax {
  constructor(depth = 3) {
    this.depth = depth;
    this.name = 'Minimax d=' + depth;
  }

  getBestMove(game) {
    const valid = game.getValidCols();
    if (!valid.length) return 0;
    // Instant win check
    for (const col of valid) {
      const g = game.clone(); g.drop(col);
      if (g.winner === game.current) return col;
    }
    let bestScore = -Infinity, bestCol = valid[Math.floor(valid.length / 2)];
    for (const col of valid) {
      const g = game.clone(); g.drop(col);
      const score = this._minimax(g, this.depth - 1, -Infinity, Infinity, false, game.current);
      if (score > bestScore) { bestScore = score; bestCol = col; }
    }
    return bestCol;
  }

  _minimax(game, depth, alpha, beta, maximizing, player) {
    if (game.over) {
      if (game.winner === player) return 100000 + depth;
      if (game.winner !== 0) return -100000 - depth;
      return 0;
    }
    if (depth === 0) return game.scorePosition(player);
    const valid = game.getValidCols();
    if (maximizing) {
      let val = -Infinity;
      for (const col of valid) {
        const g = game.clone(); g.drop(col);
        val = Math.max(val, this._minimax(g, depth - 1, alpha, beta, false, player));
        alpha = Math.max(alpha, val);
        if (alpha >= beta) break;
      }
      return val;
    } else {
      let val = Infinity;
      for (const col of valid) {
        const g = game.clone(); g.drop(col);
        val = Math.min(val, this._minimax(g, depth - 1, alpha, beta, true, player));
        beta = Math.min(beta, val);
        if (beta <= alpha) break;
      }
      return val;
    }
  }
}

// ─────────────────────────────────────
// Neural AI player
// ─────────────────────────────────────
class Connect4NeuralAI {
  constructor(brain) { this.brain = brain; }

  getBestMove(game) {
    const valid = game.getValidCols();
    if (!valid.length) return 0;
    // Instant win / block check first (helps early in training)
    for (const col of valid) {
      const g = game.clone(); g.drop(col);
      if (g.winner === game.current) return col;
    }
    const opp = game.current === 1 ? 2 : 1;
    for (const col of valid) {
      const g = game.clone();
      g.current = opp; g.drop(col);
      if (g.winner === opp) return col;
    }
    const inputs = game.getInputs(game.current);
    const outputs = this.brain.predict(inputs);
    // Pick best valid column
    let best = valid[0], bestScore = -Infinity;
    for (const col of valid) {
      if (outputs[col] > bestScore) { bestScore = outputs[col]; best = col; }
    }
    return best;
  }
}

// ─────────────────────────────────────
// Connect4AIInstance — one trainable AI
// name, population, independent GA + training loop
// ─────────────────────────────────────
class Connect4AIInstance {
  constructor({ name = 'AI', populationSize = 30, isMaster = false, onGeneration } = {}) {
    this.name = name;
    this.isMaster = isMaster;
    this.popSize = populationSize;
    this.onGeneration = onGeneration || (() => {});
    this.generation = 0;
    this.bestEver = 0;
    this.lastBest = 0;
    this.lastAvg = 0;
    this.running = false;
    this._loopId = null;

    this.ga = new GeneticAlgorithm({
      populationSize,
      eliteCount: 3,
      mutationRate: 0.15,
      mutationStrength: 0.3,
    });
    this.ga.init(() => new NeuralNetwork(43, [32, 16], 7));

    // Opponents at increasing difficulty
    this._opps = [
      new Connect4Minimax(1),
      new Connect4Minimax(2),
      new Connect4Minimax(3),
    ];

    // Watch game state — one game being played for display
    this.watchGame = new Connect4Game();
    this._watchAI = new Connect4NeuralAI(this.ga.population[0].clone());
    this._watchOpp = this._opps[0];
    this._watchTurn = 0; // steps taken in watch game
  }

  // Evaluate entire population (synchronous, fast)
  _evalGeneration() {
    const scores = [];
    for (let i = 0; i < this.popSize; i++) {
      const ai = new Connect4NeuralAI(this.ga.population[i]);
      // Play against 3 opponents, score is sum
      let total = 0;
      for (const opp of this._opps) {
        total += this._playMatch(ai, opp);
      }
      scores.push(total);
      this.ga.setFitness(i, total);
    }
    this.lastAvg = scores.reduce((a, b) => a + b, 0) / scores.length;
    this.lastBest = Math.max(...scores);
    if (this.lastBest > this.bestEver) this.bestEver = this.lastBest;
  }

  _playMatch(ai, opp) {
    const game = new Connect4Game();
    let moves = 0;
    while (!game.over && moves < 100) {
      if (game.current === 1) {
        game.drop(ai.getBestMove(game));
      } else {
        game.drop(opp.getBestMove(game));
      }
      moves++;
    }
    if (game.winner === 1) return 200 + (42 - moves) * 2;
    if (game.winner === 2) return Math.max(0, 50 - moves);
    return 80; // draw is decent
  }

  _refreshWatchGame() {
    // Pick best brain so far for the watch game
    const brain = this.ga.bestEver || this.ga.population[0];
    this._watchAI = new Connect4NeuralAI(brain.clone());
    this._watchOpp = this._opps[Math.min(2, Math.floor(this.generation / 10))];
    this.watchGame = new Connect4Game();
  }

  // Step the watch game by one move (called by renderer)
  stepWatchGame() {
    if (this.watchGame.over) {
      this._refreshWatchGame();
      return;
    }
    if (this.watchGame.current === 1) {
      this.watchGame.drop(this._watchAI.getBestMove(this.watchGame));
    } else {
      this.watchGame.drop(this._watchOpp.getBestMove(this.watchGame));
    }
  }

  start() {
    if (this.running) return;
    this.running = true;
    this._tick();
  }

  pause() {
    this.running = false;
    if (this._loopId) { clearTimeout(this._loopId); this._loopId = null; }
  }

  toggle() { if (this.running) this.pause(); else this.start(); }

  _tick() {
    if (!this.running) return;
    this._evalGeneration();
    this.ga.evolve();
    this.generation = this.ga.generation;
    this.onGeneration({
      generation: this.generation,
      best: this.lastBest,
      avg: this.lastAvg,
      bestEver: this.bestEver,
    });
    // Schedule next generation — yield to browser between generations
    this._loopId = setTimeout(() => this._tick(), 0);
  }

  getBestBrain() { return this.ga.bestEver; }
}

// ─────────────────────────────────────
// Connect4Renderer
// ─────────────────────────────────────
class Connect4Renderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    canvas.width  = C4_COLS * C4_CELL + 2;
    canvas.height = C4_ROWS * C4_CELL + 2;
  }

  draw(game, hoverCol = -1) {
    const ctx = this.ctx;
    const W = this.canvas.width;
    const H = this.canvas.height;

    ctx.fillStyle = '#1a1a8a';
    ctx.beginPath();
    ctx.roundRect(0, 0, W, H, 8);
    ctx.fill();

    for (let r = 0; r < C4_ROWS; r++) {
      for (let c = 0; c < C4_COLS; c++) {
        const x = c * C4_CELL + 1;
        const y = r * C4_CELL + 1;
        const cx = x + C4_CELL / 2;
        const cy = y + C4_CELL / 2;
        const radius = C4_CELL / 2 - 5;

        if (c === hoverCol && game.board[0][c] === 0) {
          ctx.fillStyle = 'rgba(255,255,255,0.1)';
          ctx.fillRect(x, 0, C4_CELL, H);
        }

        const cell = game.board[r][c];
        const inWin = game.winLine && game.winLine.some(([wr, wc]) => wr === r && wc === c);

        ctx.beginPath();
        ctx.arc(cx, cy, radius, 0, Math.PI * 2);

        if (cell === 0)      ctx.fillStyle = '#080810';
        else if (cell === 1) ctx.fillStyle = inWin ? '#ff0055' : '#ff4466';
        else                 ctx.fillStyle = inWin ? '#ffdd00' : '#ffd700';
        ctx.fill();

        if (inWin) {
          ctx.strokeStyle = '#fff';
          ctx.lineWidth = 3;
          ctx.stroke();
        }
      }
    }

    if (game.over) {
      ctx.fillStyle = 'rgba(0,0,0,0.6)';
      ctx.fillRect(0, H / 2 - 30, W, 60);
      ctx.fillStyle = '#fff';
      ctx.font = 'bold 22px sans-serif';
      ctx.textAlign = 'center';
      const msg = game.winner === 0 ? 'Draw' : (game.winner === 1 ? 'AI wins' : 'Opponent wins');
      ctx.fillText(msg, W / 2, H / 2 + 8);
      ctx.textAlign = 'left';
    }
  }
}

// Exports
window.Connect4Game       = Connect4Game;
window.Connect4Minimax    = Connect4Minimax;
window.Connect4NeuralAI   = Connect4NeuralAI;
window.Connect4AIInstance = Connect4AIInstance;
window.Connect4Renderer   = Connect4Renderer;
window.C4_COLS = C4_COLS;
window.C4_ROWS = C4_ROWS;
window.C4_CELL = C4_CELL;
