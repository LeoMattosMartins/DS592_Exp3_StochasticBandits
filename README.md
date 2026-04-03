# DS592 Exp3 on Stochastic Bandits

## Overview
This repository contains the programming assignment for computing and comparing the behavior of **Upper Confidence Bound (UCB)** and **Exponential-weight algorithm for Exploration and Exploitation (Exp3)** on stochastic multi-armed bandits. 

Specifically, this assignment tracks expected regret for a two-armed stochastic Bernoulli bandit instance. The models investigate performance over varying horizons, sensitivity to Exp3's learning rate parameters, tolerance for differing true reward gaps ($\Delta$), and an evaluation of minimax worst-case outcomes against theoretical adversarial guarantees.

## Features
- **High Performance Core**: Implementation is fully powered by **JAX** (`jax.numpy`, `jax.vmap`, `jax.lax.scan`), guaranteeing extremely fast parallel and hardware-accelerated executions, significantly reducing deep simulation times of continuous $10^5$ loop horizons down to mere seconds.
- **Reproducibility**: Contains rigorous simulations repeating over multiple algorithmic trials per horizon string.
- **Narrative Analysis**: Includes a comprehensive written conclusion reflecting on the empirical findings—interpreting the "paranoia" of adversarial algorithms versus the confidence of stochastic algorithms.

## Quickstart & Dependencies

You can recreate the environment rapidly using [uv](https://docs.astral.sh/uv/):
```bash
uv venv .venv
source .venv/bin/activate
uv pip install jupyter numpy matplotlib tqdm jax jaxlib jupytext ipykernel
```

To enable using this as a jupyter kernel:
```bash
python -m ipykernel install --user --name=ds592_exp3_bandits --display-name "Python (DS592 Bandits)"
```

## Repository Structure
- `bandit_experiments.ipynb`: The main executable notebook containing all logic, simulated algorithms, plotted visual outputs, and final conclusions.

## Experiments Summary
- **Experiment A**: Benchmarks cumulative regret against changing total horizon limits $n$.
- **Experiment B**: Isolates the learning rate $\eta \in [0.001, 0.1]$ and explores how tuning impacts ultimate expected regret over $n=100,000$ fixed periods.
- **Experiment C**: Iterates the logic from (B) over varying stochastic gap difficulties ($\Delta \in \{0.01, 0.05, 0.1, 0.2, 0.3\}$).
- **Experiment D**: Minimizes worst-case bounds. Identifies the empirical $\eta$ that minimizes maximum potential loss, discovering that an empirically tuned $\eta_{emp} \approx 0.0078$ can afford to be more aggressive than purely defensive theoretic bounds ($\eta \approx 0.0026$) when competing against fair systems.
- **Experiment E (Conclusion)**: A narrative translating the numeric plots into a story characterizing UCB as a confident scientist, and Exp3 as a paranoid gambler.
