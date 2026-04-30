/**
 * CONNECT 4 — Game Engine + AI (Minimax + Neural Net)
 */

const C4_COLS = 7;
const C4_ROWS = 6;
const C4_CELL = 64;

class Connect4Game {
  constructor() { this.reset(); }

  reset() {
    // 0 = empty, 1 = player1 (red), 2 = player2 (yellow/AI)
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
    if (this.over || this.board[0][col] !== 0) return false;
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
          this.over = true; // draw
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
    g.winLine = this.winLine;
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

  // Neural net inputs: flatten board (42) + current player (1) = 43 inputs
  getInputs(forPlayer = 1) {
    const inputs = [];
    for (let r = 0; r < C4_ROWS; r++) {
      for (let c = 0; c < C4_COLS; c++) {
        const cell = this.board[r][c];
        inputs.push(cell === forPlayer ? 1 : cell === 0 ? 0 : -1);
      }
    }
    inputs.push(this.current === forPlayer ? 1 : -1);
    return inputs; // 43 inputs
  }

  // Score position heuristically for minimax
  scorePosition(player) {
    let score = 0;
    const opp = player === 1 ? 2 : 1;

    // Center preference
    const center = this.board.map(r => r[Math.floor(C4_COLS / 2)]);
    score += center.filter(c => c === player).length * 3;

    // Horizontal
    for (let r = 0; r < C4_ROWS; r++) {
      for (let c = 0; c <= C4_COLS - 4; c++) {
        const window = [this.board[r][c], this.board[r][c+1], this.board[r][c+2], this.board[r][c+3]];
        score += this._scoreWindow(window, player, opp);
      }
    }
    // Vertical
    for (let c = 0; c < C4_COLS; c++) {
      for (let r = 0; r <= C4_ROWS - 4; r++) {
        const window = [this.board[r][c], this.board[r+1][c], this.board[r+2][c], this.board[r+3][c]];
        score += this._scoreWindow(window, player, opp);
      }
    }
    // Diagonals
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
// Minimax AI (classic, depth-limited)
// ─────────────────────────────────────
class Connect4Minimax {
  constructor(depth = 5) {
    this.depth = depth;
    this.name = `Minimax (d=${depth})`;
  }

  getBestMove(game) {
    let bestScore = -Infinity;
    let bestCol = game.getValidCols()[0];
    for (const col of game.getValidCols()) {
      const clone = game.clone();
      clone.drop(col);
      const score = this._minimax(clone, this.depth - 1, -Infinity, Infinity, false, game.current);
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

    const validCols = game.getValidCols();
    if (maximizing) {
      let val = -Infinity;
      for (const col of validCols) {
        const clone = game.clone();
        clone.drop(col);
        val = Math.max(val, this._minimax(clone, depth - 1, alpha, beta, false, player));
        alpha = Math.max(alpha, val);
        if (alpha >= beta) break;
      }
      return val;
    } else {
      let val = Infinity;
      for (const col of validCols) {
        const clone = game.clone();
        clone.drop(col);
        val = Math.min(val, this._minimax(clone, depth - 1, alpha, beta, true, player));
        beta = Math.min(beta, val);
        if (beta <= alpha) break;
      }
      return val;
    }
  }
}

// ─────────────────────────────────────
// Neural net AI for Connect 4
// Uses GA to train against minimax
// ─────────────────────────────────────
class Connect4NeuralAI {
  constructor(brain) {
    this.brain = brain;
  }

  getBestMove(game) {
    const valid = game.getValidCols();
    const inputs = game.getInputs(game.current);
    const outputs = this.brain.predict(inputs);
    // Pick highest output among valid cols
    let best = valid[0];
    let bestScore = -Infinity;
    for (const col of valid) {
      if (outputs[col] > bestScore) { bestScore = outputs[col]; best = col; }
    }
    return best;
  }
}

// ─────────────────────────────────────
// Connect4Trainer — trains neural AI via GA
// ─────────────────────────────────────
class Connect4Trainer {
  constructor({ populationSize = 50, onGeneration } = {}) {
    this.popSize = populationSize;
    this.onGeneration = onGeneration || (() => {});
    this.ga = new GeneticAlgorithm({
      populationSize,
      eliteCount: 3,
      mutationRate: 0.12,
      mutationStrength: 0.25
    });
    this.ga.init(() => new NeuralNetwork(43, [32, 16], 7));
    this.generation = 0;
    this.bestScore = 0;
    this.bestEver = 0;
    this.avgScore = 0;
    this.running = false;
    this.speed = 1;
    this._opponents = [
      new Connect4Minimax(1),
      new Connect4Minimax(2),
      new Connect4Minimax(3),
    ];
    this._currentIdx = 0;
    this.watchGame = null;
    this._evalAll();
  }

  _evalAll() {
    // Each agent plays against progressively harder opponents
    const scores = this.ga.population.map((brain, i) => {
      const ai = new Connect4NeuralAI(brain);
      const opp = this._opponents[Math.floor(i / this.popSize * this._opponents.length)];
      return this._playMatch(ai, opp);
    });
    scores.forEach((s, i) => this.ga.setFitness(i, s));

    this.avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;
    this.bestScore = Math.max(...scores);
    if (this.bestScore > this.bestEver) this.bestEver = this.bestScore;

    // Set a watch game for visualization (best brain vs strongest minimax)
    const best = this.ga.population[scores.indexOf(this.bestScore)];
    this.watchGame = new Connect4Game();
    this._watchBrain = new Connect4NeuralAI(best);
    this._watchOpp = new Connect4Minimax(3);
    this._watchTurn = 0; // 0 = neural AI goes first
  }

  _playMatch(ai, opp) {
    const game = new Connect4Game();
    let moves = 0;
    while (!game.over && moves < 100) {
      if (game.current === 1) {
        // AI is player 1
        const col = ai.getBestMove(game);
        game.drop(col);
      } else {
        const col = opp.getBestMove(game);
        game.drop(col);
      }
      moves++;
    }
    if (game.winner === 1) return 100 + (42 - moves);
    if (game.winner === 2) return Math.max(0, 20 - moves);
    return 30; // draw is ok
  }

  start() { this.running = true; this._trainLoop(); }
  pause() { this.running = false; }
  toggle() { if (this.running) this.pause(); else this.start(); }
  setSpeed(s) { this.speed = s; }

  _trainLoop() {
    if (!this.running) return;
    this.ga.evolve();
    this.generation = this.ga.generation;
    this._evalAll();
    this.onGeneration({ generation: this.generation, best: this.bestScore, avg: this.avgScore, bestEver: this.bestEver });
    // Use setTimeout to not block UI
    setTimeout(() => this._trainLoop(), 10);
  }

  // Advance watch game one move
  stepWatchGame() {
    if (!this.watchGame || this.watchGame.over) {
      // Restart
      this.watchGame = new Connect4Game();
    }
    const game = this.watchGame;
    if (game.current === 1 && this._watchBrain) {
      const col = this._watchBrain.getBestMove(game);
      game.drop(col);
    } else if (this._watchOpp) {
      const col = this._watchOpp.getBestMove(game);
      game.drop(col);
    }
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
    canvas.width = C4_COLS * C4_CELL + 2;
    canvas.height = C4_ROWS * C4_CELL + 2;
  }

  draw(game, hoverCol = -1) {
    const ctx = this.ctx;
    const W = this.canvas.width;
    const H = this.canvas.height;

    // Background
    ctx.fillStyle = '#1a1a8a';
    ctx.roundRect(0, 0, W, H, 8);
    ctx.fill();

    for (let r = 0; r < C4_ROWS; r++) {
      for (let c = 0; c < C4_COLS; c++) {
        const x = c * C4_CELL + 1;
        const y = r * C4_CELL + 1;
        const cx = x + C4_CELL / 2;
        const cy = y + C4_CELL / 2;
        const radius = C4_CELL / 2 - 5;

        // Hover highlight
        if (c === hoverCol && game.board[0][c] === 0) {
          ctx.fillStyle = 'rgba(255,255,255,0.1)';
          ctx.fillRect(x, 0, C4_CELL, H);
        }

        ctx.beginPath();
        ctx.arc(cx, cy, radius, 0, Math.PI * 2);

        const cell = game.board[r][c];

        // Check if this cell is in the win line
        const inWin = game.winLine && game.winLine.some(([wr, wc]) => wr === r && wc === c);

        if (cell === 0) {
          ctx.fillStyle = '#080810';
        } else if (cell === 1) {
          ctx.fillStyle = inWin ? '#ff0055' : '#ff4466';
        } else {
          ctx.fillStyle = inWin ? '#ffdd00' : '#ffd700';
        }
        ctx.fill();

        // Win glow
        if (inWin) {
          ctx.strokeStyle = '#fff';
          ctx.lineWidth = 3;
          ctx.stroke();
        }
      }
    }

    // Status overlay
    if (game.over) {
      ctx.fillStyle = 'rgba(0,0,0,0.6)';
      ctx.fillRect(0, H / 2 - 30, W, 60);
      ctx.fillStyle = '#fff';
      ctx.font = 'bold 22px sans-serif';
      ctx.textAlign = 'center';
      const msg = game.winner === 0 ? 'Draw!' : `Player ${game.winner} wins!`;
      ctx.fillText(msg, W / 2, H / 2 + 8);
      ctx.textAlign = 'left';
    }
  }
}

// Exports
window.Connect4Game = Connect4Game;
window.Connect4Minimax = Connect4Minimax;
window.Connect4NeuralAI = Connect4NeuralAI;
window.Connect4Trainer = Connect4Trainer;
window.Connect4Renderer = Connect4Renderer;
window.C4_COLS = C4_COLS;
window.C4_ROWS = C4_ROWS;
window.C4_CELL = C4_CELL;
