/**
 * PONG — Synthwave Neon Edition
 * Co-evolution: two separate populations (left vs right)
 *
 * Architecture matches Snake/Connect4:
 *   - tick() called every RAF frame, runs trainSpeed generations
 *   - Watch game advances separately via stepWatchGame() on an interval
 */

const PW = 600;
const PH = 400;
const PADDLE_W = 10;
const PADDLE_H = 70;
const BALL_SIZE = 8;
const PADDLE_SPEED = 4;
const MAX_SCORE = 5;

// ─────────────────────────────────────
// PongGame
// ─────────────────────────────────────
class PongGame {
  constructor() { this.reset(); }

  reset() {
    this.leftY   = PH / 2 - PADDLE_H / 2;
    this.rightY  = PH / 2 - PADDLE_H / 2;
    this.ballX   = PW / 2;
    this.ballY   = PH / 2;
    const angle  = (Math.random() * Math.PI / 3) - Math.PI / 6;
    const dir    = Math.random() < 0.5 ? 1 : -1;
    this.ballVX  = dir * 5 * Math.cos(angle);
    this.ballVY  = 5 * Math.sin(angle);
    this.scoreL  = 0;
    this.scoreR  = 0;
    this.rallies = 0;
    this.steps   = 0;
    this.over    = false;
    this.trail   = [];
  }

  getInputs(side) {
    const paddleY = side === 'left' ? this.leftY  : this.rightY;
    const oppY    = side === 'left' ? this.rightY : this.leftY;
    return [
      this.ballX  / PW,
      this.ballY  / PH,
      this.ballVX / 10,
      this.ballVY / 10,
      (paddleY + PADDLE_H / 2) / PH,
      (oppY    + PADDLE_H / 2) / PH,
    ];
  }

  moveLeft(action)  { this.leftY  = Math.max(0, Math.min(PH - PADDLE_H, this.leftY  + action * PADDLE_SPEED)); }
  moveRight(action) { this.rightY = Math.max(0, Math.min(PH - PADDLE_H, this.rightY + action * PADDLE_SPEED)); }

  step() {
    if (this.over) return;
    this.steps++;
    this.ballX += this.ballVX;
    this.ballY += this.ballVY;

    this.trail.push({ x: this.ballX, y: this.ballY });
    if (this.trail.length > 18) this.trail.shift();

    if (this.ballY <= 0)  { this.ballY = 0;  this.ballVY *= -1; }
    if (this.ballY >= PH) { this.ballY = PH; this.ballVY *= -1; }

    if (this.ballX - BALL_SIZE / 2 <= 22 + PADDLE_W &&
        this.ballX - BALL_SIZE / 2 >= 22 &&
        this.ballY >= this.leftY && this.ballY <= this.leftY + PADDLE_H) {
      this.ballVX = Math.abs(this.ballVX) * 1.03;
      this.ballVY = ((this.ballY - (this.leftY + PADDLE_H / 2)) / (PADDLE_H / 2)) * 6;
      this.rallies++;
    }

    if (this.ballX + BALL_SIZE / 2 >= PW - 22 - PADDLE_W &&
        this.ballX + BALL_SIZE / 2 <= PW - 22 &&
        this.ballY >= this.rightY && this.ballY <= this.rightY + PADDLE_H) {
      this.ballVX = -Math.abs(this.ballVX) * 1.03;
      this.ballVY = ((this.ballY - (this.rightY + PADDLE_H / 2)) / (PADDLE_H / 2)) * 6;
      this.rallies++;
    }

    const speed = Math.sqrt(this.ballVX ** 2 + this.ballVY ** 2);
    if (speed > 14) { this.ballVX *= 14 / speed; this.ballVY *= 14 / speed; }

    if (this.ballX < 0)  { this.scoreR++; this._resetBall(1);  }
    if (this.ballX > PW) { this.scoreL++; this._resetBall(-1); }

    if (this.scoreL >= MAX_SCORE || this.scoreR >= MAX_SCORE) this.over = true;
    if (this.steps > 3000) this.over = true;
  }

  _resetBall(dir) {
    this.ballX = PW / 2; this.ballY = PH / 2;
    const angle = (Math.random() * Math.PI / 3) - Math.PI / 6;
    this.ballVX = dir * 5 * Math.cos(angle);
    this.ballVY = 5 * Math.sin(angle);
    this.trail  = [];
  }

  getFitness(side) {
    const myScore  = side === 'left' ? this.scoreL : this.scoreR;
    const oppScore = side === 'left' ? this.scoreR : this.scoreL;
    return myScore * 20 + this.rallies * 3 - oppScore * 10;
  }
}

// ─────────────────────────────────────
// PongAI
// ─────────────────────────────────────
class PongAI {
  constructor(brain, side) { this.brain = brain; this.side = side; }

  act(game) {
    const outputs = this.brain.predict(game.getInputs(this.side));
    const maxIdx  = outputs.indexOf(Math.max(...outputs));
    const action  = maxIdx === 0 ? -1 : maxIdx === 2 ? 1 : 0;
    if (this.side === 'left') game.moveLeft(action);
    else                      game.moveRight(action);
  }
}

// ─────────────────────────────────────
// PongTrainer
// Matches the pattern of SnakeAIInstance / Connect4AIInstance:
//   tick()           — called every RAF frame, runs trainSpeed gens
//   stepWatchGame()  — called by a separate interval, advances the display game
// ─────────────────────────────────────
class PongTrainer {
  constructor({ populationSize = 50, onGeneration } = {}) {
    this.popSize      = populationSize;
    this.onGeneration = onGeneration || (() => {});
    this.generation   = 0;
    this.bestLeft     = 0;
    this.bestRight    = 0;
    this.bestEver     = 0;
    this.avgRallies   = 0;
    this.running      = false;
    this.trainSpeed   = 1;  // gens per RAF tick

    this.gaLeft  = new GeneticAlgorithm({ populationSize, eliteCount: 4, mutationRate: 0.12, mutationStrength: 0.25 });
    this.gaRight = new GeneticAlgorithm({ populationSize, eliteCount: 4, mutationRate: 0.12, mutationStrength: 0.25 });
    this.gaLeft.init(()  => new NeuralNetwork(6, [12, 8], 3));
    this.gaRight.init(() => new NeuralNetwork(6, [12, 8], 3));

    // Watch game — separate from training, updated by stepWatchGame()
    this.watchGame   = new PongGame();
    this._watchLeft  = new PongAI(this.gaLeft.population[0].clone(),  'left');
    this._watchRight = new PongAI(this.gaRight.population[0].clone(), 'right');

    // Win tracking
    this._winsLeft  = 0;
    this._winsRight = 0;
  }

  // Called every RAF frame — runs trainSpeed generations synchronously
  tick() {
    if (!this.running) return;
    for (let i = 0; i < this.trainSpeed; i++) {
      this._runOneGeneration();
    }
  }

  _runOneGeneration() {
    const fitL = new Array(this.popSize).fill(0);
    const fitR = new Array(this.popSize).fill(0);
    const rallies = [];

    for (let i = 0; i < this.popSize; i++) {
      const j   = Math.floor(Math.random() * this.popSize);
      const aiL = new PongAI(this.gaLeft.population[i],  'left');
      const aiR = new PongAI(this.gaRight.population[j], 'right');
      const game = new PongGame();

      while (!game.over) {
        aiL.act(game); aiR.act(game); game.step();
      }

      fitL[i] += game.getFitness('left');
      fitR[j] += game.getFitness('right');
      rallies.push(game.rallies);
    }

    fitL.forEach((f, i) => this.gaLeft.setFitness(i, f));
    fitR.forEach((f, i) => this.gaRight.setFitness(i, f));

    this.bestLeft   = Math.max(...fitL);
    this.bestRight  = Math.max(...fitR);
    this.avgRallies = rallies.reduce((a, b) => a + b, 0) / rallies.length;
    const best = Math.max(this.bestLeft, this.bestRight);
    if (best > this.bestEver) this.bestEver = best;

    this.gaLeft.evolve();
    this.gaRight.evolve();
    this.generation = this.gaLeft.generation;

    // Refresh watch agents with best brains
    const bl = this.gaLeft.bestEver  || this.gaLeft.population[0];
    const br = this.gaRight.bestEver || this.gaRight.population[0];
    this._watchLeft  = new PongAI(bl.clone(), 'left');
    this._watchRight = new PongAI(br.clone(), 'right');

    this.onGeneration({
      generation:  this.generation,
      bestLeft:    this.bestLeft,
      bestRight:   this.bestRight,
      avgRallies:  this.avgRallies,
      bestEver:    this.bestEver,
    });
  }

  // Called by a timer on the page — steps the watch game one tick
  stepWatchGame() {
    if (this.watchGame.over) {
      const w = this.watchGame.scoreL >= MAX_SCORE ? 'left'
              : this.watchGame.scoreR >= MAX_SCORE ? 'right' : null;
      if (w === 'left')  this._winsLeft++;
      if (w === 'right') this._winsRight++;
      this.watchGame = new PongGame();
      return;
    }
    this._watchLeft.act(this.watchGame);
    this._watchRight.act(this.watchGame);
    this.watchGame.step();
  }

  setTrainSpeed(s) { this.trainSpeed = Math.max(1, Math.min(20, s)); }
  start()  { this.running = true; }
  pause()  { this.running = false; }
  toggle() { this.running = !this.running; }

  getWinCounts() { return { left: this._winsLeft, right: this._winsRight }; }
  getBestBrains() {
    return {
      left:  this.gaLeft.bestEver,
      right: this.gaRight.bestEver,
    };
  }
}

// ─────────────────────────────────────
// PongRenderer — synthwave neon style
// ─────────────────────────────────────
class PongRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx    = canvas.getContext('2d');
    canvas.width  = PW;
    canvas.height = PH;
  }

  draw(game) {
    const ctx = this.ctx;

    ctx.fillStyle = '#05050f';
    ctx.fillRect(0, 0, PW, PH);

    this._drawGrid(ctx);
    this._drawScanlines(ctx);

    // Center dashed line
    ctx.setLineDash([8, 8]);
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(PW / 2, 0); ctx.lineTo(PW / 2, PH); ctx.stroke();
    ctx.setLineDash([]);

    // Scores
    ctx.font = 'bold 34px JetBrains Mono, monospace';
    ctx.textAlign = 'center';
    ctx.shadowColor = '#ff2d78'; ctx.shadowBlur = 18;
    ctx.fillStyle = '#ff2d78';
    ctx.fillText(game.scoreL, PW / 2 - 60, 48);
    ctx.shadowColor = '#00f5ff'; ctx.shadowBlur = 18;
    ctx.fillStyle = '#00f5ff';
    ctx.fillText(game.scoreR, PW / 2 + 60, 48);
    ctx.shadowBlur = 0; ctx.textAlign = 'left';

    // Ball trail
    game.trail.forEach((p, i) => {
      const alpha = (i / game.trail.length) * 0.45;
      const size  = BALL_SIZE * (i / game.trail.length) * 0.8;
      ctx.fillStyle = `rgba(255,247,0,${alpha})`;
      ctx.fillRect(p.x - size / 2, p.y - size / 2, size, size);
    });

    // Ball
    ctx.shadowColor = '#fff700'; ctx.shadowBlur = 22;
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(game.ballX - BALL_SIZE / 2, game.ballY - BALL_SIZE / 2, BALL_SIZE, BALL_SIZE);
    ctx.fillStyle = '#fff700';
    ctx.fillRect(game.ballX - 2, game.ballY - 2, 4, 4);
    ctx.shadowBlur = 0;

    // Paddles
    this._drawPaddle(ctx, 22,              game.leftY,  '#ff2d78');
    this._drawPaddle(ctx, PW - 22 - PADDLE_W, game.rightY, '#00f5ff');

    // Rally counter
    ctx.font = '11px JetBrains Mono, monospace';
    ctx.fillStyle = 'rgba(255,255,255,0.25)';
    ctx.textAlign = 'center';
    ctx.fillText('RALLY ' + game.rallies, PW / 2, PH - 8);
    ctx.textAlign = 'left';

    // Game over overlay
    if (game.over) {
      ctx.fillStyle = 'rgba(0,0,0,0.55)';
      ctx.fillRect(0, PH / 2 - 32, PW, 64);
      const winner = game.scoreL >= MAX_SCORE ? 'LEFT WINS' : game.scoreR >= MAX_SCORE ? 'RIGHT WINS' : 'DRAW';
      const color  = game.scoreL >= MAX_SCORE ? '#ff2d78' : '#00f5ff';
      ctx.shadowColor = color; ctx.shadowBlur = 28;
      ctx.fillStyle = color;
      ctx.font = 'bold 26px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      ctx.fillText(winner, PW / 2, PH / 2 + 9);
      ctx.shadowBlur = 0; ctx.textAlign = 'left';
    }
  }

  _drawPaddle(ctx, x, y, color) {
    ctx.shadowColor = color; ctx.shadowBlur = 18;
    ctx.fillStyle = color;
    ctx.fillRect(x, y, PADDLE_W, PADDLE_H);
    ctx.fillStyle = 'rgba(255,255,255,0.6)';
    ctx.fillRect(x, y, 2, PADDLE_H);
    ctx.shadowBlur = 0;
    ctx.fillStyle = 'rgba(255,255,255,0.15)';
    for (let i = 6; i < PADDLE_H; i += 10) ctx.fillRect(x + 3, y + i, 4, 2);
  }

  _drawGrid(ctx) {
    const horizon = PH * 0.55;
    ctx.strokeStyle = 'rgba(180,0,255,0.1)'; ctx.lineWidth = 1;
    for (let i = 0; i < 8; i++) {
      const t = i / 7, y = horizon + (PH - horizon) * t;
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(PW, y); ctx.stroke();
    }
    for (let i = 0; i <= 12; i++) {
      const x = (i / 12) * PW;
      ctx.beginPath();
      ctx.moveTo(PW / 2 + (x - PW / 2) * 0.05, horizon);
      ctx.lineTo(x, PH);
      ctx.stroke();
    }
  }

  _drawScanlines(ctx) {
    ctx.fillStyle = 'rgba(0,0,0,0.06)';
    for (let y = 0; y < PH; y += 3) ctx.fillRect(0, y, PW, 1);
  }
}

window.PongGame     = PongGame;
window.PongAI       = PongAI;
window.PongTrainer  = PongTrainer;
window.PongRenderer = PongRenderer;
window.PW = PW; window.PH = PH;
