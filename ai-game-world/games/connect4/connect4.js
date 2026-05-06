/**
 * CONNECT 4 — Game Engine + AI
 * Neural net trained via Genetic Algorithm against Minimax opponents
 *
 * Architecture:
 *   tick()           — called every RAF frame. Uses a time budget (12ms max)
 *                      so it never blocks rendering. Runs full generations.
 *   stepWatchGame()  — called by a setInterval, advances display game at human pace.
 *   autosave         — every 50 generations, posts best brain to Supabase.
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
    this.board     = Array.from({ length: C4_ROWS }, () => new Array(C4_COLS).fill(0));
    this.current   = 1;
    this.winner    = 0;
    this.over      = false;
    this.lastMove  = null;
    this.winLine   = null;
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
        this.lastMove  = { row, col };
        this.moveCount++;
        const win = this.checkWin(row, col, this.current);
        if (win) {
          this.winner  = this.current;
          this.winLine = win;
          this.over    = true;
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
    const g     = new Connect4Game();
    g.board     = this.board.map(r => [...r]);
    g.current   = this.current;
    g.winner    = this.winner;
    g.over      = this.over;
    g.lastMove  = this.lastMove  ? { ...this.lastMove }             : null;
    g.moveCount = this.moveCount;
    g.winLine   = this.winLine   ? this.winLine.map(x => [...x])    : null;
    return g;
  }

  checkWin(row, col, player) {
    const dirs = [[0,1],[1,0],[1,1],[1,-1]];
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
  constructor(depth = 2) { this.depth = depth; }

  getBestMove(game) {
    const valid = game.getValidCols();
    if (!valid.length) return 0;
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
      if (game.winner !== 0)      return -100000 - depth;
      return 0;
    }
    if (depth === 0) return game.scorePosition(player);
    const valid = game.getValidCols();
    if (maximizing) {
      let val = -Infinity;
      for (const col of valid) {
        const g = game.clone(); g.drop(col);
        val   = Math.max(val, this._minimax(g, depth - 1, alpha, beta, false, player));
        alpha = Math.max(alpha, val);
        if (alpha >= beta) break;
      }
      return val;
    } else {
      let val = Infinity;
      for (const col of valid) {
        const g = game.clone(); g.drop(col);
        val  = Math.min(val, this._minimax(g, depth - 1, alpha, beta, true, player));
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
    // Instant win
    for (const col of valid) {
      const g = game.clone(); g.drop(col);
      if (g.winner === game.current) return col;
    }
    // Block opponent win
    const opp = game.current === 1 ? 2 : 1;
    for (const col of valid) {
      const g = game.clone(); g.current = opp; g.drop(col);
      if (g.winner === opp) return col;
    }
    const outputs = this.brain.predict(game.getInputs(game.current));
    let best = valid[0], bestScore = -Infinity;
    for (const col of valid) {
      if (outputs[col] > bestScore) { bestScore = outputs[col]; best = col; }
    }
    return best;
  }
}

// ─────────────────────────────────────
// Connect4AIInstance
// ─────────────────────────────────────
class Connect4AIInstance {
  constructor({ name = 'AI', populationSize = 30, isMaster = false, onGeneration, supabaseUrl, supabaseKey } = {}) {
    this.name         = name;
    this.isMaster     = isMaster;
    this.popSize      = populationSize;
    this.onGeneration = onGeneration || (() => {});
    this.generation   = 0;
    this.bestEver     = 0;
    this.lastBest     = 0;
    this.lastAvg      = 0;
    this.running      = false;
    this.trainSpeed   = 1;
    this._supabaseUrl = supabaseUrl || null;
    this._supabaseKey = supabaseKey || null;

    this.ga = new GeneticAlgorithm({
      populationSize,
      eliteCount:       3,
      mutationRate:     0.15,
      mutationStrength: 0.3,
    });
    this.ga.init(() => new NeuralNetwork(43, [32, 16], 7));

    this._opps = [
      new Connect4Minimax(1),
      new Connect4Minimax(2),
      new Connect4Minimax(3),
    ];

    this.watchGame  = new Connect4Game();
    this._watchAI   = new Connect4NeuralAI(this.ga.population[0]);
    this._watchOpp  = this._opps[0];
  }

  _runOneGeneration() {
    for (let i = 0; i < this.popSize; i++) {
      const ai    = new Connect4NeuralAI(this.ga.population[i]);
      let total   = 0;
      for (const opp of this._opps) total += this._playMatch(ai, opp);
      this.ga.setFitness(i, total);
    }

    const scores  = this.ga.fitnesses;
    this.lastAvg  = scores.reduce((a, b) => a + b, 0) / scores.length;
    this.lastBest = Math.max(...scores);
    if (this.lastBest > this.bestEver) this.bestEver = this.lastBest;

    this.ga.evolve();
    this.generation = this.ga.generation;

    const brain      = this.ga.bestEver || this.ga.population[0];
    this._watchAI    = new Connect4NeuralAI(brain);
    this._watchOpp   = this._opps[Math.min(2, Math.floor(this.generation / 8))];

    // Autosave every 50 generations
    if (this.isMaster && this.generation % 50 === 0 && this._supabaseUrl && this._supabaseKey) {
      this._autosave(brain);
    }

    this.onGeneration({
      generation: this.generation,
      best:       this.lastBest,
      avg:        this.lastAvg,
      bestEver:   this.bestEver,
    });
  }

  _autosave(brain) {
    const payload = {
      game:       'connect4',
      generation: this.generation,
      best_score: this.bestEver,
      avg_score:  parseFloat(this.lastAvg.toFixed(2)),
      brain_json: JSON.stringify(brain.toJSON()),
      saved_at:   new Date().toISOString(),
    };
    fetch(this._supabaseUrl + '/rest/v1/master_ai', {
      method:  'POST',
      headers: {
        'Content-Type':  'application/json',
        'apikey':         this._supabaseKey,
        'Authorization': 'Bearer ' + this._supabaseKey,
        'Prefer':        'resolution=merge-duplicates',
      },
      body: JSON.stringify(payload),
    }).catch(() => {});
  }

  _playMatch(ai, opp) {
    const game = new Connect4Game();
    let moves  = 0;
    while (!game.over && moves < 50) {
      if (game.current === 1) game.drop(ai.getBestMove(game));
      else                    game.drop(opp.getBestMove(game));
      moves++;
    }
    if (game.winner === 1) return 200 + (42 - moves) * 2;
    if (game.winner === 2) return Math.max(0, 50 - moves);
    return 80;
  }

  // Called every RAF frame — time-budgeted so it never blocks rendering
  tick() {
    if (!this.running) return;
    const deadline = performance.now() + 12; // 12ms budget
    for (let i = 0; i < this.trainSpeed; i++) {
      if (performance.now() > deadline) break;
      this._runOneGeneration();
    }
  }

  stepWatchGame() {
    if (this.watchGame.over) {
      this.watchGame = new Connect4Game();
      return;
    }
    if (this.watchGame.current === 1) this.watchGame.drop(this._watchAI.getBestMove(this.watchGame));
    else                              this.watchGame.drop(this._watchOpp.getBestMove(this.watchGame));
  }

  refreshWatchGame() {
    const brain   = this.ga.bestEver || this.ga.population[0];
    this._watchAI = new Connect4NeuralAI(brain);
    this.watchGame = new Connect4Game();
  }

  setTrainSpeed(s) { this.trainSpeed = Math.max(1, Math.min(20, s)); }
  start()  { this.running = true; }
  pause()  { this.running = false; }
  toggle() { this.running = !this.running; }
  getBestBrain() { return this.ga.bestEver; }
}

// ─────────────────────────────────────
// Connect4Renderer
// ─────────────────────────────────────
class Connect4Renderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx    = canvas.getContext('2d');
    canvas.width  = C4_COLS * C4_CELL;
    canvas.height = C4_ROWS * C4_CELL;
  }

  draw(game, hoverCol = -1) {
    const ctx = this.ctx;
    const W   = C4_COLS * C4_CELL;
    const H   = C4_ROWS * C4_CELL;
    const R   = C4_CELL / 2 - 4;

    // Background
    ctx.fillStyle = '#060612';
    ctx.fillRect(0, 0, W, H);

    // Board
    ctx.fillStyle = '#0d1230';
    ctx.fillRect(0, 0, W, H);

    // Hover highlight
    if (hoverCol >= 0) {
      ctx.fillStyle = 'rgba(91,124,250,0.07)';
      ctx.fillRect(hoverCol * C4_CELL, 0, C4_CELL, H);
    }

    // Cells
    for (let r = 0; r < C4_ROWS; r++) {
      for (let c = 0; c < C4_COLS; c++) {
        const cx = c * C4_CELL + C4_CELL / 2;
        const cy = r * C4_CELL + C4_CELL / 2;
        const cell = game.board[r][c];

        // Empty hole
        ctx.fillStyle = '#080810';
        ctx.beginPath();
        ctx.arc(cx, cy, R, 0, Math.PI * 2);
        ctx.fill();

        if (cell === 1) {
          ctx.shadowColor = '#f05068'; ctx.shadowBlur = 12;
          ctx.fillStyle   = '#f05068';
        } else if (cell === 2) {
          ctx.shadowColor = '#f5c842'; ctx.shadowBlur = 12;
          ctx.fillStyle   = '#f5c842';
        }

        if (cell !== 0) {
          ctx.beginPath();
          ctx.arc(cx, cy, R, 0, Math.PI * 2);
          ctx.fill();
          ctx.shadowBlur = 0;

          // Shine
          ctx.fillStyle = 'rgba(255,255,255,0.2)';
          ctx.beginPath();
          ctx.arc(cx - R * 0.25, cy - R * 0.28, R * 0.35, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }

    // Win line highlight
    if (game.winLine) {
      ctx.strokeStyle = 'rgba(255,255,255,0.85)';
      ctx.lineWidth   = 4;
      ctx.shadowColor = 'white'; ctx.shadowBlur = 16;
      ctx.setLineDash([8, 4]);
      game.winLine.forEach(([r, c], i) => {
        const cx = c * C4_CELL + C4_CELL / 2;
        const cy = r * C4_CELL + C4_CELL / 2;
        i === 0 ? ctx.beginPath() && ctx.moveTo(cx, cy) : null;
        if (i === 0) { ctx.beginPath(); ctx.moveTo(cx, cy); }
        else           ctx.lineTo(cx, cy);
      });
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.shadowBlur = 0;
    }

    // Grid lines
    ctx.strokeStyle = 'rgba(255,255,255,0.04)';
    ctx.lineWidth   = 1;
    for (let c = 1; c < C4_COLS; c++) {
      ctx.beginPath(); ctx.moveTo(c * C4_CELL, 0); ctx.lineTo(c * C4_CELL, H); ctx.stroke();
    }
    for (let r = 1; r < C4_ROWS; r++) {
      ctx.beginPath(); ctx.moveTo(0, r * C4_CELL); ctx.lineTo(W, r * C4_CELL); ctx.stroke();
    }

    // Last move indicator
    if (game.lastMove) {
      const { row, col } = game.lastMove;
      ctx.strokeStyle = 'rgba(255,255,255,0.4)';
      ctx.lineWidth   = 2;
      ctx.beginPath();
      ctx.arc(col * C4_CELL + C4_CELL / 2, row * C4_CELL + C4_CELL / 2, R + 4, 0, Math.PI * 2);
      ctx.stroke();
    }
  }
}

// Export
window.Connect4Game       = Connect4Game;
window.Connect4Minimax    = Connect4Minimax;
window.Connect4NeuralAI   = Connect4NeuralAI;
window.Connect4AIInstance = Connect4AIInstance;
window.Connect4Renderer   = Connect4Renderer;
window.C4_COLS = C4_COLS;
window.C4_ROWS = C4_ROWS;
window.C4_CELL = C4_CELL;
