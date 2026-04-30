/**
 * NEAT-inspired Neural Network + Genetic Algorithm
 * Lightweight, browser-compatible, no dependencies
 */

class NeuralNetwork {
  constructor(inputSize, hiddenSizes, outputSize) {
    this.inputSize = inputSize;
    this.hiddenSizes = hiddenSizes;
    this.outputSize = outputSize;
    this.layers = [];
    this.biases = [];

    // Build layers
    const sizes = [inputSize, ...hiddenSizes, outputSize];
    for (let i = 0; i < sizes.length - 1; i++) {
      const layer = [];
      for (let j = 0; j < sizes[i + 1]; j++) {
        const neuron = [];
        for (let k = 0; k < sizes[i]; k++) {
          neuron.push((Math.random() * 2 - 1));
        }
        layer.push(neuron);
      }
      this.layers.push(layer);
      this.biases.push(new Array(sizes[i + 1]).fill(0).map(() => (Math.random() * 2 - 1)));
    }
  }

  activate(x) {
    return Math.tanh(x); // tanh activation
  }

  predict(inputs) {
    let current = inputs;
    for (let l = 0; l < this.layers.length; l++) {
      const next = [];
      for (let j = 0; j < this.layers[l].length; j++) {
        let sum = this.biases[l][j];
        for (let k = 0; k < current.length; k++) {
          sum += this.layers[l][j][k] * current[k];
        }
        next.push(this.activate(sum));
      }
      current = next;
    }
    return current;
  }

  clone() {
    const nn = new NeuralNetwork(this.inputSize, [...this.hiddenSizes], this.outputSize);
    nn.layers = JSON.parse(JSON.stringify(this.layers));
    nn.biases = JSON.parse(JSON.stringify(this.biases));
    return nn;
  }

  mutate(rate = 0.1, strength = 0.2) {
    for (let l = 0; l < this.layers.length; l++) {
      for (let j = 0; j < this.layers[l].length; j++) {
        for (let k = 0; k < this.layers[l][j].length; k++) {
          if (Math.random() < rate) {
            this.layers[l][j][k] += (Math.random() * 2 - 1) * strength;
            this.layers[l][j][k] = Math.max(-2, Math.min(2, this.layers[l][j][k]));
          }
        }
        if (Math.random() < rate) {
          this.biases[l][j] += (Math.random() * 2 - 1) * strength;
        }
      }
    }
    return this;
  }

  crossover(other) {
    const child = this.clone();
    for (let l = 0; l < child.layers.length; l++) {
      for (let j = 0; j < child.layers[l].length; j++) {
        for (let k = 0; k < child.layers[l][j].length; k++) {
          if (Math.random() < 0.5) {
            child.layers[l][j][k] = other.layers[l][j][k];
          }
        }
        if (Math.random() < 0.5) {
          child.biases[l][j] = other.biases[l][j];
        }
      }
    }
    return child;
  }

  toJSON() {
    return { inputSize: this.inputSize, hiddenSizes: this.hiddenSizes, outputSize: this.outputSize, layers: this.layers, biases: this.biases };
  }

  static fromJSON(data) {
    const nn = new NeuralNetwork(data.inputSize, data.hiddenSizes, data.outputSize);
    nn.layers = data.layers;
    nn.biases = data.biases;
    return nn;
  }
}

class GeneticAlgorithm {
  constructor({ populationSize = 50, eliteCount = 5, mutationRate = 0.1, mutationStrength = 0.3 } = {}) {
    this.populationSize = populationSize;
    this.eliteCount = eliteCount;
    this.mutationRate = mutationRate;
    this.mutationStrength = mutationStrength;
    this.generation = 0;
    this.population = [];
    this.fitnesses = [];
    this.bestFitness = 0;
    this.bestEver = null;
    this.avgFitness = 0;
    this.history = [];
  }

  init(createGenome) {
    this.population = Array.from({ length: this.populationSize }, () => createGenome());
    this.fitnesses = new Array(this.populationSize).fill(0);
  }

  setFitness(index, fitness) {
    this.fitnesses[index] = fitness;
  }

  evolve() {
    // Sort by fitness
    const ranked = this.fitnesses
      .map((f, i) => ({ f, i }))
      .sort((a, b) => b.f - a.f);

    this.avgFitness = this.fitnesses.reduce((a, b) => a + b, 0) / this.fitnesses.length;
    const topFitness = ranked[0].f;

    if (topFitness > this.bestFitness) {
      this.bestFitness = topFitness;
      this.bestEver = this.population[ranked[0].i].clone();
    }

    this.history.push({ gen: this.generation, best: topFitness, avg: this.avgFitness });
    if (this.history.length > 200) this.history.shift();

    const newPop = [];

    // Elites survive
    for (let i = 0; i < this.eliteCount; i++) {
      newPop.push(this.population[ranked[i].i].clone());
    }

    // Fill rest with crossover + mutation
    while (newPop.length < this.populationSize) {
      const p1 = this.tournamentSelect(ranked);
      const p2 = this.tournamentSelect(ranked);
      const child = p1.crossover(p2).mutate(this.mutationRate, this.mutationStrength);
      newPop.push(child);
    }

    this.population = newPop;
    this.fitnesses = new Array(this.populationSize).fill(0);
    this.generation++;
  }

  tournamentSelect(ranked, k = 5) {
    let best = null;
    for (let i = 0; i < k; i++) {
      const idx = ranked[Math.floor(Math.random() * ranked.length)].i;
      if (!best || this.fitnesses[idx] > this.fitnesses[best]) best = idx;
    }
    return this.population[best || ranked[0].i];
  }
}

// Export for modules or global
if (typeof module !== 'undefined') {
  module.exports = { NeuralNetwork, GeneticAlgorithm };
} else {
  window.NeuralNetwork = NeuralNetwork;
  window.GeneticAlgorithm = GeneticAlgorithm;
}
